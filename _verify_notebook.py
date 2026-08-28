import json

with open(r'D:\Unsloth\TESI_FINALE\benchmark.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Valid notebook:', nb['nbformat'], '.', nb['nbformat_minor'])
print('Total cells:', len(nb['cells']))
print()
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell['source'])[:80].replace('\n', ' ')
    print(f'  {i:2d}. {ctype:10s} | {src}')
