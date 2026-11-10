"""HTTP value server for one trained critic (LoRA "critic" + value head, as trained by train_ppo_alf.py); used by the
critic rules of ct_infer.py and by offline_collect.py. POST /value {"ids": [[token ids], ...]} -> {"v": [floats]};  GET /health.
Concurrent requests are gathered by one GPU thread and scored together, packed into sequences of at most --pack-tokens
tokens (flash-attention keeps the records apart through position_ids that restart at 0)."""
import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--max-len', type=int, default=8192)
    ap.add_argument('--pack-tokens', type=int, default=0,
                    help='0: one sequence per forward (bit-identical to unbatched scoring; sibling value gaps are often at the '
                         'size of bf16 differences between packed and unpacked forwards)')
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation='flash_attention_2').cuda()
    model = PeftModel.from_pretrained(model, os.path.join(a.critic, 'critic'), adapter_name='critic')
    model.set_adapter('critic'); model.eval()
    vhead = torch.nn.Linear(model.config.hidden_size, 1).cuda().float()
    vhead.load_state_dict(torch.load(os.path.join(a.critic, 'value_head.pt'), map_location='cuda'))
    inner = model.base_model.model.model
    pending = queue.Queue()

    @torch.no_grad()
    def packed_values(seqs):
        out = [None] * len(seqs)

        def run(pack):
            ids, pos, last = [], [], []
            for _, x in pack:
                ids += x; pos += list(range(len(x))); last.append(len(ids) - 1)
            h = inner(input_ids=torch.tensor([ids], device='cuda'),
                      position_ids=torch.tensor([pos], device='cuda')).last_hidden_state[0, last]
            for (i, _), v in zip(pack, vhead(h.float())[:, 0].tolist()):
                out[i] = v
        pack, n = [], 0
        for i, x in enumerate(seqs):
            x = x[-a.max_len:]
            if pack and n + len(x) > max(a.pack_tokens, 1):
                run(pack); pack, n = [], 0
            pack.append((i, x)); n += len(x)
        if pack:
            run(pack)
        return out

    def gpu_loop():
        while True:
            items = [pending.get()]
            time.sleep(0.005)
            while True:
                try:
                    items.append(pending.get_nowait())
                except queue.Empty:
                    break
            try:
                vals = packed_values([x for it in items for x in it['ids']])
                k = 0
                for it in items:
                    it['v'] = vals[k:k + len(it['ids'])]; k += len(it['ids'])
            except Exception as e:  # noqa: BLE001
                for it in items:
                    it['err'] = repr(e)
            for it in items:
                it['ev'].set()
    threading.Thread(target=gpu_loop, daemon=True).start()

    def values(batch):
        it = {'ids': batch, 'ev': threading.Event()}
        pending.put(it); it['ev'].wait()
        if 'err' in it:
            raise RuntimeError(it['err'])
        return it['v']

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            self._send(200, {'ok': True})

        def do_POST(self):
            n = int(self.headers.get('Content-Length', 0))
            req = json.loads(self.rfile.read(n))
            try:
                self._send(200, {'v': values(req['ids'])})
            except Exception as e:  # noqa: BLE001
                self._send(500, {'error': repr(e)})
    print(f'value server for {a.critic} on :{a.port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', a.port), H).serve_forever()


if __name__ == '__main__':
    main()
