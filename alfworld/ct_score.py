"""Pairwise test, step 2: score every parent and child state of ct_collect.py's output with a trained critic
(the LoRA "critic" adapter + value head that train_ppo_alf.py trains; last hidden state of the turn prompt).
Writes <in>.scored.json with v_parent per state and v per candidate (None for terminal candidates)."""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--critic', required=True, help='<it>/critic directory (critic/ adapter + value_head.pt)')
    ap.add_argument('--base', required=True)
    ap.add_argument('--max-len', type=int, default=8192)
    ap.add_argument('--tag', default='', help='output <in>.scored<_tag>.json (another critic on the same states)')
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda()
    model = PeftModel.from_pretrained(model, os.path.join(a.critic, 'critic'), adapter_name='critic')
    model.set_adapter('critic'); model.eval()
    vhead = torch.nn.Linear(model.config.hidden_size, 1).cuda().float()
    vhead.load_state_dict(torch.load(os.path.join(a.critic, 'value_head.pt'), map_location='cuda'))
    inner = model.base_model.model.model

    @torch.no_grad()
    def V(ids):
        h = inner(input_ids=torch.tensor([ids[-a.max_len:]], device='cuda')).last_hidden_state[0, -1]
        return float(vhead(h.float())[0])

    d = json.load(open(a.data))
    n = 0
    for s in d['states']:
        if len(s.get('cands', [])) < 2:
            continue
        s['v_parent'] = V(s['parent_prompt_ids'])
        for c in s['cands']:
            c['v'] = V(c['child_prompt_ids']) if (c['child_prompt_ids'] and not c['terminal']) else None
            n += 1
    for s in d['states']:                       # keep the output small
        s.pop('parent_prompt_ids', None)
        for c in s.get('cands', []):
            c.pop('child_prompt_ids', None)
    out = a.data.replace('.json', f".scored{'_' + a.tag if a.tag else ''}.json")
    json.dump(d, open(out, 'w'), indent=1)
    print(f'scored {n} candidates -> {out}', flush=True)


if __name__ == '__main__':
    main()
