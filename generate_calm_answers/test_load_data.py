from datasets import load_from_disk
data = load_from_disk('./squad_v2_unanswerable_with_generated_answers')

print(data)
if 'train' in data:
    print(len(data['train']))
if 'validation' in data:
    print(len(data['validation']))

print("\n" + "="*80)
print("Individual elements of the unanswerable validation set:")
print("="*80 + "\n")

for i, item in enumerate(data['validation']):
    print(f"Element {i}:")
    print(item)
    print("-"*80)