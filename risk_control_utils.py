import numpy as np
from typing import List, Tuple, Dict
from scipy.stats import binom, entropy
from scipy.optimize import brentq
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

def apply_risk_control(conf, acc, gt, rel_labels, lambdas, eps_grid, delta, n_cal, n_trials, loss_01_conversion, default_to_zero_shot=True, lambda_type='confidence'): # lambda_type = 'confidence' or 'layer'

    if lambda_type == 'confidence':
        losses, exits = get_losses_and_exits_confidence(conf, acc, lambdas, rel_labels, gt, default_to_zero_shot)
    else:
        losses, exits = get_losses_and_exits_layer(conf, acc, lambdas, rel_labels, gt)

    # Compute risk and efficiency gains & optimal lambdas (using rc_main)
    avg_exit_per_lambda = exits.mean(axis=1)
    rcp_type = 'ltt-scaled' if loss_01_conversion == 'scaling' else 'ltt'
    test_risk, eff_gains, rcp_lams = rc_main(np.array(losses), avg_exit_per_lambda, np.array(eps_grid), rcp_types=[rcp_type],
                                             loss_bound=1, n_trials=n_trials, n_cal=n_cal, delta=delta, binary_loss=True,)
    test_risk, eff_gains, rcp_lams = np.array(test_risk[rcp_type]), np.array(eff_gains[rcp_type]), np.array(rcp_lams[rcp_type])
    return losses, test_risk, eff_gains, rcp_lams


def get_relative_labels(relative_labels, correct, incorrect, zeroshot, n_early_exit):
    all_layers = []
    full_model_idx = str(n_early_exit-1)
    if relative_labels == 'full_model':
        return correct[full_model_idx], incorrect[full_model_idx], zeroshot[full_model_idx]
    elif relative_labels == 'zeroshot_full_model':
        return zeroshot[full_model_idx], zeroshot[full_model_idx], zeroshot[full_model_idx]
    else:
        # don't compute relative loss
        return None, None, None

def load_all_data(base_dir):
    if 'correct' not in os.listdir(base_dir) or 'incorrect' not in os.listdir(base_dir) or 'zeroshot' not in os.listdir(base_dir):
        return 'Missing some data'
        
    c_files = [f for f in os.listdir(base_dir + 'correct/') if os.path.isfile(os.path.join(base_dir + 'correct/', f))]
    i_files = [f for f in os.listdir(base_dir + 'incorrect/') if os.path.isfile(os.path.join(base_dir + 'incorrect/', f))]
    z_files = [f for f in os.listdir(base_dir + 'zeroshot/') if os.path.isfile(os.path.join(base_dir + 'zeroshot/', f))]
    if len(c_files) == 0 or len(i_files) == 0 or len(z_files) == 0:
        return 'Missing some data'
        
    # Load data
    with open(base_dir + 'correct/' + c_files[0], 'r') as file:
        correct = json.load(file)

    with open(base_dir + 'incorrect/' + c_files[0], 'r') as file:
        incorrect = json.load(file)

    with open(base_dir + 'zeroshot/' + c_files[0], 'r') as file:
        zeroshot = json.load(file)

    return correct, incorrect, zeroshot


def get_ground_truth_by_type(ground_truth_type, correct, incorrect, zeroshot, n_early_exit):
    full_model_idx = str(n_early_exit-1)
    if ground_truth_type == 'true_label':
        return correct['true_label'], incorrect['true_label'], zeroshot['true_label']
    elif ground_truth_type == 'zeroshot_full_model':
        return zeroshot[full_model_idx], zeroshot[full_model_idx], zeroshot[full_model_idx]
    elif ground_truth_type == 'full_model':
        return correct[full_model_idx], incorrect[full_model_idx], zeroshot[full_model_idx]


# Lambda exit layers MUST BE in decreasing order
def get_losses_and_exits_layer(conf, acc, lambda_exit_layers, relative_labels, true_labels, default_to_zero_shot=True):
    # Track losses and exit layers
    all_losses, all_exits = [], []

    for layer in lambda_exit_layers: # iterate over exit layers
        # Compute all losses via accuracy
        losses = 1-acc[:,layer]
        # Exits are all the same layer; Convert to 1-based indexing
        exits = (layer+1) * np.ones(losses.shape)
        all_losses.append(losses)
        all_exits.append(exits)

    # If relative loss, compute
    if relative_labels is not None:
        relative_labels_loss = [1 - (x == y) for x,y in zip(relative_labels, true_labels)]
        # If default to zero-shot, add another exit option with the zero-shot loss
        if default_to_zero_shot:
            all_losses.append(relative_labels_loss)
            all_exits.append((lambda_exit_layers[0]+1) * np.ones(losses.shape))
        all_losses = np.array(all_losses) - np.array(relative_labels_loss)

    return np.array(all_losses), np.array(all_exits)


# Lambdas MUST BE in decreasing order
def get_losses_and_exits_confidence(conf, acc, lambdas, relative_labels, true_labels, default_to_zero_shot=True):
    # Track losses and exit layers
    all_losses, all_exits = [], []
    true_labels = np.array(true_labels)
    if relative_labels is not None:
        relative_labels = np.array(relative_labels)

    for lam_idx in range(len(lambdas)): # iterate over lambdas, 1 to 0
        l = lambdas[lam_idx]
        # Find the first exit exceeding lambda in confidence
        mask = (conf > l).astype(np.int32)
        exits = np.argmax(mask, axis=1)
        # Whenever the threshold is not crossed, exit at the final layer
        rows_with_no_threshold = np.sum(mask, axis=1) == 0
        exits[rows_with_no_threshold] = mask.shape[1] - 1
        # convert to 1-based indexing for exits (instead of 0-based)
        exits = exits + 1
        # Compute all losses via accuracy
        lambda_acc = acc[np.arange(len(exits)), exits - 1] # accuracy at each of the chosen exit points

        if default_to_zero_shot:
            lambda_acc[rows_with_no_threshold] = (relative_labels[rows_with_no_threshold] == true_labels[rows_with_no_threshold])

        all_exits.append(exits)
        losses = (1 - lambda_acc)
        all_losses.append(losses)

    # If relative loss, compute
    if relative_labels is not None:
        relative_labels_loss = [1 - (x == y) for x,y in zip(relative_labels, true_labels)]
        all_losses = np.array(all_losses) - np.array(relative_labels_loss)

    return np.array(all_losses), np.array(all_exits)

def get_all_accuracies(data, true_labels, n_early_exit, first_exit=0):
    acc = np.zeros((len(data['0']), n_early_exit-first_exit))
    for data_idx in range(len(data['0'])):
        for layer in range(first_exit, n_early_exit):
            # Get the prediction from this layer
            pred, true_label = data[str(layer)][data_idx], true_labels[data_idx]
            acc[data_idx][layer-first_exit] = 1 if pred == true_label else 0
    return acc

# compute normalized entropy in range [0,1]
# then compute 1 - that to get the confidence
def compute_entropy_confidence(prob_dist):
    prob_dist = np.array(prob_dist)
    prob_dist = prob_dist[prob_dist > 0]  # filter out zero probabilities to avoid log(0)
    H = entropy(prob_dist, base=2)
    max_H = np.log2(len(prob_dist))
    normalized_entropy = H / max_H if max_H > 0 else 0.0
    return 1 - normalized_entropy


def get_all_confidences(data, n_early_exit, label_order, confidence_type, first_exit=0):
    conf = np.zeros((len(data['0']), n_early_exit-first_exit))
    for data_idx in range(len(data['0'])):
        for layer in range(first_exit, n_early_exit):
            # Get confidences from this layer
            confidences = data[str(layer) + '_confidences'][data_idx]

            # Based on the confidence type, compute and save confidence
            # argmax or top2_diff or entropy
            layer_idx = layer - first_exit
            if confidence_type == 'argmax':
                # Pick the confidence corresponding to the predicted answer and add it to the array
                pred_label_idx = label_order.index(data[str(layer)][data_idx])
                conf[data_idx][layer_idx] = confidences[pred_label_idx]
            elif confidence_type == 'top2_diff':
                # Pick the top 2 elements of the confidences and add their difference to the array
                top2, _ = torch.topk(torch.Tensor(confidences),2)
                conf[data_idx][layer_idx] = top2[0] - top2[1]
            elif confidence_type == 'entropy':
                # Compute entropy over the entire confidences array
                conf[data_idx][layer_idx] = compute_entropy_confidence(confidences)
    return conf


def get_label_order(dataset, tokenizer_name):
    with open('dataset_labels.json') as f:
        dataset_labels = json.load(f)
    
    with open('all_token_maps.json') as f:
        token_maps = json.load(f)
    token_map = token_maps[tokenizer_name]

    # Reduce down to only the allowed labels
    keys_to_remove = []
    for key in token_map:
        if key not in dataset_labels[dataset]:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        token_map.pop(key, None)

    return list(token_map.keys())


def rc_main(
    losses: np.array,
    exits: np.array,
    eps_grid: np.array,
    rcp_types: List[str],
    binary_loss: bool = False,
    loss_bound: int = 1,
    n_trials: int = 5,
    n_cal: int = 100,
    delta: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Compute the upper risk controlling exiting threshold.

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        exits: (K,) average exit per threshold
        eps_grid: grid of risk levels
        rcp_types: list of risk control procedures
        binary_loss: whether the loss is binary
        loss_bound: upper bound on the loss
        n_trials: number of trials (calibration/test splits)
        n_cal: number of calibration datapoints
        delta: confidence level
        seed: random seed

    """
    np.random.seed(seed)

    for x in rcp_types:
        assert x in ["naive", "ltt", "ucb-wsr", "crc", "ltt-scaled"]

    assert losses.shape[0] == exits.shape[0]
    _, N = losses.shape

    test_risk, eff_gains = {r: [] for r in rcp_types}, {r: [] for r in rcp_types}

    for _ in range(n_trials):
        # select n_cal datapoints from N
        cal_ids = np.random.choice(N, n_cal, replace=False)
        test_ids = np.setdiff1d(np.arange(N), cal_ids)

        cal_losses, test_losses = losses[:, cal_ids], losses[:, test_ids]

        # STAGE 1: find \hat{\lambda} on the calibration dataset
        rcp_lams = {r: [] for r in rcp_types}
        for rcp in rcp_types:
            for eps in eps_grid:
                if rcp == "naive":
                    lam_id = naive_lam(cal_losses, eps)
                elif rcp == "ucb-wsr":
                    lam_id = ucb_lam(
                        cal_losses, eps, delta, ucb_type="wsr", binary_loss=False, loss_bound=loss_bound
                    )
                elif rcp == "ltt":
                    lam_id = ltt_lam(
                        np.maximum(cal_losses, 0.0), eps, delta, binary_loss
                    )
                    # print(rcp, eps, lam_id)
                elif rcp == "ltt-scaled":
                    A = -1
                    B = 1
                    lam_id = ltt_lam(
                        (cal_losses - A) / (B - A), (eps - A) / (B - A), delta, binary_loss
                    )
                    # print(rcp, eps, lam_id)
                elif rcp == "crc":
                    lam_id = crc_lam(cal_losses, eps, loss_bound=loss_bound)
                rcp_lams[rcp].append(lam_id)

        # STAGE 2: using \hat{\lambda} from the stage 1 to compute test risk and efficiency gains
        for rcp in rcp_types:
            test_risk_e, eff_gains_e = [], []
            for e, eps in enumerate(eps_grid):
                lam_id = rcp_lams[rcp][e]
                test_risk_e.append(test_losses[lam_id].mean())
                eff_gains_e.append(exits[lam_id])
            test_risk[rcp].append(test_risk_e)
            eff_gains[rcp].append(eff_gains_e)

    return test_risk, eff_gains, rcp_lams


def naive_lam(losses: np.array, epsilon: float) -> int:
    """
    Find a naive lambda (Eq. 9)
    
    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
    
    """

    risk = losses.mean(axis=1)

    lams = (risk < epsilon).nonzero()[0]

    if len(lams) == 0:
        return None
    else:
        return lams.max()


def crc_lam(losses: np.array, epsilon: float, loss_bound: float = 1.0) -> int:
    """
    Find risk controlling lambda based on Conformal Risk Control (CRC, Eq. 10)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        loss_bound: upper bound on the loss
    
    """
    _, n_cal = losses.shape

    risk = losses.mean(axis=1)

    ucb = (n_cal + 1) * epsilon / n_cal - loss_bound / n_cal

    lams = (risk <= ucb).nonzero()[0]

    if len(lams) == 0:
        return None
    else:
        return lams.max()

def ltt_lam(
    losses: np.array, epsilon: float, delta: float, binary_loss: bool
) -> int:
    """
    Find risk controlling lambda based on Learn-then-Test (LTT)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        delta: confidence level
        binary_loss: whether the loss is binary
    """
    K, n_cal = losses.shape
    risk = losses.mean(axis=1)

    p_vals = []
    for k in range(K):
        p_vals.append(
            hb_p_value(
                risk=risk[k], n=n_cal, alpha=epsilon, binary_loss=binary_loss
            ).item()
        )

    lams = (np.array(p_vals) <= delta).nonzero()[0]
    if len(lams) == 0:
        return 0
    else:
        return lams.max()


def ucb_lam(
    losses: np.array,
    epsilon: float,
    delta: float,
    ucb_type: str = "wsr",
    binary_loss: bool = False,
    loss_bound: float = 1,
) -> int:
    """
    Find risk-controlling lambda using upper confidence bound (UCB) calibration (Eq. 12)

    Args:
        losses: (K, n_cal) where K is the size of the lambda grid. 
                 Rows in losses array correspond to the descending ordering of lambdas.
        epsilon: tolerated risk level 
        delta: confidence level
        ucb_type: type of UCB. Options: "wsr", "hb"
        binary_loss: whether the loss is binary
        loss_bound: upper bound on the loss
    
    """
    K, n_cal = losses.shape

    ucb = []
    for k in range(K):
        if ucb_type == "wsr":
            ucb.append(ucb_wsr(losses[k], delta=delta, B=loss_bound))
        elif ucb_type == "hb":
            ucb.append(
                ucb_hb(
                    risk=losses[k].mean(),
                    delta=delta,
                    n_cal=n_cal,
                    binary_loss=binary_loss,
                )
            )
        else:
            raise ValueError(f"Invalid UCB type: {ucb_type}")

    lams = (np.array(ucb) < epsilon).nonzero()[0]

    # find the smallest lambda for which the UCB for all the larger lambdas is smaller than epsilon (Eq. 12)
    rc_lam = find_first_gap(lams)
    return rc_lam


def find_first_gap(arr: np.array) -> int:
    if len(arr) == 0:
        return 0
    for i in range(len(arr)):
        if arr[i] != i:
            return max(0, i - 1)
    return len(arr) - 1


# adapted from https://github.com/aangelopoulos/rcps/blob/main/core/bounds.py
def ucb_wsr(x, delta, maxiters=1000, B=1, eps=1e-10):
    """
    Compute the upper confidence bound (UCB) based on the Waudby-Smith Ramdas (WSR) bound.

    Args:
        TODO

    Returns:
        TODO
    """
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1, n + 1)))
    sigma2hat = (np.cumsum((x - muhat) ** 2) + 0.25) / (1 + np.array(range(1, n + 1)))
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log(1 / delta) / n / sigma2hat), 1 / B)

    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)

    if _Kn(1) < 0:
        return B
    if _Kn(eps) > 0:
        return eps
    return brentq(_Kn, eps, 1 - eps, maxiter=maxiters)


def ucb_hb(risk, delta, n_cal, binary_loss, step=0.01):
    """
    Compute the upper confidence bound (UCB) based on the Hoeffding-Bentkus (HB) bound.

    Args:
        TODO

    Returns:
        TODO
    """
    alphas = np.arange(0.01, 1.0 + step, step)[::-1]
    for i in range(len(alphas)):
        if (
            hb_p_value(risk=risk, n=n_cal, alpha=alphas[i], binary_loss=binary_loss)
            >= delta
        ):
            return alphas[i]
    return 0.0


# adapted from https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py
def hb_p_value(
    risk: float,
    n: int,
    alpha: float = 0.05,  # this is actually epsilon from risk control
    eps: float = 1e-3,  # this is only for numerical stability
    binary_loss: bool = False,
):
    """
    Compute the p-value of the Hoeffding-Bentkus bound.

    Args:
        risk: Computed risk estimate.
        n: Number of calibration samples.
        alpha: Tolerated risk level.

    Returns:
        p-value.
    """
    if binary_loss:
        p_value = binom.cdf(np.ceil(n * risk), n, alpha)
    else:
        bentkus_p_value = np.e * binom.cdf(np.ceil(n * risk), n, alpha)
        a, b = min(risk, alpha), alpha
        h1 = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
        hoeffding_p_value = np.exp(-n * h1)
        p_value = min(bentkus_p_value, hoeffding_p_value)

    assert 0 - eps <= p_value <= 1 + eps, "p-value must be in [0, 1]: {}".format(
        p_value
    )
    return p_value
