from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
data=yaml.safe_load((root/'DESIGN.md').read_text().split('---')[1])
lines=['/* Generated from DESIGN.md by scripts/design-tokens.py. */',':root {']
for group,prefix in [('colors','color'),('rounded','radius'),('spacing','space')]:
    for key,value in data[group].items():
        lines.append(f'  --{prefix}-{key.lower()}: {value};')
for key,value in data['typography'].items():
    lines.append(f'  --font-{key}: {value["fontFamily"]};')
lines.append('}')
(root/'src/tokens.css').write_text('\n'.join(lines)+'\n')

