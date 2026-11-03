#!/usr/bin/env python3
"""
Cache Guard stage 2: a tiny HTTP service that scores whether two prompts ask the same thing.

  POST /score {"a": "...", "b": "..."}  ->  {"score": 0.93, "ms": 4.6}

Usage:  python3 server.py --model cross-encoder/quora-distilroberta-base --port 8099
Then set "stage2_url": "http://127.0.0.1:8099/score" in the cacheguard plugin config.
"""
import argparse, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="cross-encoder/quora-distilroberta-base")
ap.add_argument("--port", type=int, default=8099)
args = ap.parse_args()

import torch
from sentence_transformers import CrossEncoder
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = CrossEncoder(args.model, device=device)
model.predict([("warm up", "warm up")], show_progress_bar=False)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path != "/score":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        t0 = time.perf_counter()
        score = float(model.predict([(body.get("a", ""), body.get("b", ""))], show_progress_bar=False)[0])
        out = json.dumps({"score": score, "ms": round((time.perf_counter() - t0) * 1000, 2)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


print(f"cacheguard stage2: {args.model} on {device}, listening on 127.0.0.1:{args.port}")
ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()
