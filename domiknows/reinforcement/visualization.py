"""Flask web visualizer for reward-driven (reinforcement) training.

`ReinforcementVisualizer` attaches a ``step_hook`` to a
:class:`~domiknows.reinforcement.reinforcement_program.ReinforcementProgram`.  On
every training step the program hands the visualizer a JSON-safe payload
describing that step — the decoding (predicted per-instance distributions), the
sampled decodings, the decoded "generated sample", the applied reward, and the
calculated loss — and the visualizer **blocks the training thread** until you
advance it from the browser.  So when the visualization is activated it controls
step progression.

It is generic: it works for any ``ReinforcementProgram`` 
without changes, because all the per-step detail comes from the program's hook.

Flask is imported lazily (only when the server starts), so importing this module
does not require Flask to be installed.

Usage::

    from domiknows.reinforcement import ReinforcementVisualizer
    viz = ReinforcementVisualizer(port=5000).attach(program).start()
    program.train(dataset, train_epoch_num=5, Optim=...)

or simply ``ReinforcementProgram(..., visualize=True)``.
"""

import threading
import time
import webbrowser

__all__ = ["ReinforcementVisualizer", "VisualizationStopped"]


class VisualizationStopped(BaseException):
    """Raised inside the training thread when the user clicks *Stop*.

    Subclasses ``BaseException`` (like ``KeyboardInterrupt``) so it is not
    swallowed by ``except Exception`` handlers and propagates up to abort
    training.  ``ReinforcementProgram.train`` catches it.
    """


class ReinforcementVisualizer:
    """Serve a step-by-step view of reinforcement training and gate progression.

    :param host: bind host (default ``127.0.0.1``).
    :param port: bind port (default ``5000``).
    :param auto_open: open the dashboard in a browser when ``start()`` is called.
    :param default_mode: ``'step'`` (pause every step) or ``'play'`` (auto-advance).
    :param delay: seconds between steps while in ``'play'`` mode.
    :param exit_on_stop: when ``True`` (default), clicking *Stop* exits the whole
        program (``sys.exit``) after aborting training; set ``False`` to merely
        stop training and let ``program.train`` return.
    """

    def __init__(self, host="127.0.0.1", port=5000, auto_open=True,
                 default_mode="step", delay=0.5, exit_on_stop=True):
        self.host = host
        self.port = port
        self.auto_open = auto_open
        self.exit_on_stop = exit_on_stop

        self._cond = threading.Condition()
        self._state = None            # latest step payload
        self._status = "idle"         # idle | waiting | running | done | stopped
        self._mode = default_mode     # step | play
        self._delay = float(delay)
        self._advance_requested = False
        self._stop = False
        self._step_count = 0

        self._started = False
        self._app = None
        self._server_thread = None

    # ------------------------------------------------------------------
    # Program-facing API
    # ------------------------------------------------------------------
    def attach(self, program):
        """Route a program's per-step payloads to this visualizer.

        Also registers itself on the program so the program can notify the
        visualizer when training finishes (``mark_done``).
        """
        program.step_hook = self.step_hook
        program._visualizer = self
        return self

    def step_hook(self, payload):
        """Called by the program once per training step (may block).

        Raises :class:`VisualizationStopped` if the user requested *Stop*, which
        aborts training.
        """
        with self._cond:
            self._state = payload
            self._step_count += 1
            if self._stop:
                self._status = "stopped"
                raise VisualizationStopped()
            if self._mode == "play":
                self._status = "running"
            else:
                self._status = "waiting"
                # Block until the UI advances, switches to play, or stops.
                while (not self._advance_requested and self._mode != "play"
                       and not self._stop):
                    self._cond.wait()
                if self._stop:
                    self._status = "stopped"
                    raise VisualizationStopped()
                self._advance_requested = False
                self._status = "running"
        if self._mode == "play" and not self._stop:
            time.sleep(self._delay)

    def mark_done(self):
        """Mark training finished so the dashboard shows completion."""
        with self._cond:
            if self._status != "stopped":
                self._status = "done"
            self._cond.notify_all()

    def mark_stopped(self):
        """Mark training stopped by the user."""
        with self._cond:
            self._status = "stopped"
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------
    def start(self):
        if self._started:
            return self
        self._app = self._build_app()
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._server_thread.start()
        self._started = True
        url = f"http://{self.host}:{self.port}/"
        print(f"[ReinforcementVisualizer] serving at {url}\n"
              f"[ReinforcementVisualizer] training pauses on each step until you "
              f"click 'Next step' (or 'Play').")
        if self.auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        time.sleep(0.4)  # let the server bind before training starts blocking
        return self

    def _serve(self):
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._app.run(host=self.host, port=self.port,
                      threaded=True, use_reloader=False, debug=False)

    def _build_app(self):
        from flask import Flask, jsonify, request, Response

        app = Flask(__name__)

        @app.route("/")
        def index():
            return Response(_HTML, mimetype="text/html")

        @app.route("/api/state")
        def state():
            with self._cond:
                return jsonify({
                    "status": self._status,
                    "mode": self._mode,
                    "delay": self._delay,
                    "step_count": self._step_count,
                    "state": self._state,
                })

        @app.route("/api/next", methods=["POST"])
        def next_step():
            with self._cond:
                self._mode = "step"
                self._advance_requested = True
                self._cond.notify_all()
            return jsonify({"ok": True})

        @app.route("/api/play", methods=["POST"])
        def play():
            data = request.get_json(silent=True) or {}
            with self._cond:
                self._mode = "play"
                if "delay" in data:
                    try:
                        self._delay = max(0.0, float(data["delay"]))
                    except (TypeError, ValueError):
                        pass
                self._cond.notify_all()
            return jsonify({"ok": True})

        @app.route("/api/pause", methods=["POST"])
        def pause():
            with self._cond:
                self._mode = "step"
                self._cond.notify_all()
            return jsonify({"ok": True})

        @app.route("/api/stop", methods=["POST"])
        def stop():
            with self._cond:
                self._stop = True
                self._status = "stopped"
                self._cond.notify_all()
            return jsonify({"ok": True})

        return app


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DomiKnowS — Reinforcement training</title>
<style>
  :root{
    --bg:#0f1420; --panel:#171d2b; --panel2:#1e2638; --ink:#e7ecf5; --muted:#94a3b8;
    --line:#2a3346; --accent:#6ea8fe; --good:#34d399; --bad:#f87171; --warn:#fbbf24;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,Arial}
  header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,#121826,#0f1420);
    border-bottom:1px solid var(--line);padding:12px 18px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
  h1{font-size:15px;margin:0;font-weight:650;letter-spacing:.2px}
  .pill{padding:3px 10px;border-radius:999px;font-size:12px;border:1px solid var(--line);background:var(--panel2)}
  .pill.waiting{color:var(--warn);border-color:#5b4a1f}
  .pill.running{color:var(--accent);border-color:#2c4673}
  .pill.done{color:var(--good);border-color:#1f5b46}
  .pill.idle{color:var(--muted)}
  .spacer{flex:1}
  button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:8px;
    padding:7px 14px;font-size:13px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.primary{background:#23406e;border-color:#33598f}
  button.danger{background:#4a1d1d;border-color:#7f1d1d;color:#fca5a5}
  button.danger:hover{border-color:var(--bad)}
  button:disabled{opacity:.45;cursor:not-allowed}
  main{padding:18px;display:grid;grid-template-columns:1fr;gap:16px;max-width:1200px;margin:0 auto}
  .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
  .metric{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .metric .k{color:var(--muted);font-size:12px}
  .metric .v{font-size:22px;font-weight:680;margin-top:2px;overflow-wrap:anywhere;word-break:break-word;line-height:1.2}
  .metric .v.small{font-size:15px}
  .v.loss{color:var(--accent)} .v.reward{color:var(--good)}
  section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  section h2{margin:0 0 10px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px}
  table{border-collapse:collapse;width:100%;font-size:13px}
  th,td{border-bottom:1px solid var(--line);padding:6px 9px;text-align:left;white-space:nowrap}
  th{color:var(--muted);font-weight:600}
  .scroll{max-height:340px;overflow:auto;border:1px solid var(--line);border-radius:8px}
  .mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
  .bar{display:inline-block;height:9px;border-radius:4px;background:var(--accent);vertical-align:middle}
  .barwrap{display:inline-block;width:64px;background:#0c1018;border-radius:4px;margin-right:6px;vertical-align:middle}
  .reward-1{color:var(--good);font-weight:700}
  .reward-0{color:var(--bad)}
  tr.good td{background:rgba(52,211,153,.07)}
  .tag{display:inline-block;padding:1px 7px;border-radius:6px;background:var(--panel2);border:1px solid var(--line);margin:1px;font-size:12px}
  .tag.pos{color:var(--good);border-color:#1f5b46}
  .tag.dec{color:var(--accent);border-color:#2c4673}
  .tag.match{color:var(--good);border-color:#1f5b46}
  tr.decode-row td{background:rgba(110,168,254,.10)}
  td.cell{text-align:center}
  .colhint{color:var(--muted);font-size:11px}
  .muted{color:var(--muted)}
  .di{display:flex;gap:8px;flex-wrap:wrap}
  .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  input[type=range]{accent-color:var(--accent)}
  .hint{color:var(--muted);font-size:12px}
  .formula{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;
    color:var(--ink);font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12.5px;line-height:1.7;white-space:pre-wrap}
  .formula b{color:var(--accent)} .formula .r{color:var(--good)}
</style>
</head>
<body>
<header>
  <h1>DomiKnowS · Reinforcement training</h1>
  <span id="status" class="pill idle">idle</span>
  <span id="stepinfo" class="muted"></span>
  <span class="spacer"></span>
  <div class="controls">
    <button id="btnNext" class="primary">Next step ▶</button>
    <button id="btnPlay">Play ⏩</button>
    <button id="btnPause">Pause ⏸</button>
    <button id="btnStop" class="danger">Stop ⏹</button>
    <label class="hint">delay <input id="delay" type="range" min="0" max="2" step="0.1" value="0.5"/> <span id="delayv">0.5s</span></label>
  </div>
</header>
<main>
  <div class="metrics" id="metrics"></div>
  <div id="formula" class="formula"></div>
  <section><h2>Decoding — class ↔ instance mapping (this step)</h2><div id="targets"></div></section>
  <section><h2>Decoding → sampled decodings (this step, instance-aligned)</h2><div id="samples"></div></section>
  <section id="disec" style="display:none"><h2>Data item — input for this step</h2>
    <div class="hint" style="margin-bottom:8px">The raw training example for this step. The model reads it (via ReaderSensors) to produce the <b>decoding</b> above; for per-question rewards, its <span class="mono">reward_function</span> / <span class="mono">logic_str</span> / <span class="mono">logic_label</span> define how each sampled decoding is scored into a <b>reward</b>.</div>
    <div id="di" class="di"></div></section>
</main>
<script>
let busy=false;
const $=s=>document.querySelector(s);

async function post(url,body){ try{ await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):null}); }catch(e){} }
$('#btnNext').onclick=()=>post('/api/next');
$('#btnPlay').onclick=()=>post('/api/play',{delay:parseFloat($('#delay').value)});
$('#btnPause').onclick=()=>post('/api/pause');
$('#btnStop').onclick=()=>{ if(confirm('Stop training and exit the program?')){ post('/api/stop'); } };
$('#delay').oninput=e=>{ $('#delayv').textContent=(+e.target.value).toFixed(1)+'s'; };

function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function bar(p){const w=Math.round(p*64);return `<span class="barwrap"><span class="bar" style="width:${w}px"></span></span>`;}

function argmaxIdx(row){let m=0;row.forEach((v,j)=>{if(v>row[m])m=j;});return m;}

function renderTargets(targets){
  if(!targets||!targets.length){$('#targets').innerHTML='<span class="muted">—</span>';return;}
  let html='';
  for(const t of targets){
    html+=`<div style="margin-bottom:14px">
      <div style="margin-bottom:6px"><b>${esc(t.concept)}</b> <span class="muted">— ${t.n_instances} instance(s) × ${t.n_classes} class(es): ${t.class_names.map(esc).join(', ')}; each instance maps to its argmax class</span></div>
      <div class="scroll" style="max-height:240px"><table><thead><tr><th>instance</th>`;
    for(const cn of t.class_names) html+=`<th>P(${esc(cn)})</th>`;
    html+='<th>→ decoded class</th></tr></thead><tbody>';
    t.probabilities.forEach((row,i)=>{
      const am=argmaxIdx(row);
      html+=`<tr><td class="muted mono">${i}</td>`;
      row.forEach((v,j)=>{ const hi=j===am?'style="color:var(--good);font-weight:700"':''; html+=`<td ${hi}>${bar(v)}<span class="mono">${v.toFixed(3)}</span></td>`; });
      html+=`<td><span class="tag dec">${esc(t.class_names[am])}</span> <span class="muted mono">${row[am].toFixed(2)}</span></td>`;
      html+='</tr>';
    });
    html+='</tbody></table></div></div>';
  }
  $('#targets').innerHTML=html;
}

// Per target: a matrix whose columns are instances and whose first row is the
// decoding (argmax); each following row is one sampled decoding aligned to the
// same instances, with a check where the sample matches the decoded class.
function renderSamples(samples, targets){
  if(!samples||!samples.length||!targets||!targets.length){$('#samples').innerHTML='<span class="muted">—</span>';return;}
  let html='';
  for(const t of targets){
    const decoded=t.probabilities.map(r=>t.class_names[argmaxIdx(r)]);
    html+=`<div style="margin-bottom:16px">
      <div style="margin-bottom:6px"><b>${esc(t.concept)}</b> <span class="colhint">— classes: ${t.class_names.map(esc).join(', ')}; columns = instances 0…${t.n_instances-1}; first row = decoding, then ${samples.length} sampled decodings</span></div>
      <div class="scroll" style="max-height:300px"><table><thead><tr><th>row \\ instance</th>`;
    for(let i=0;i<t.n_instances;i++) html+=`<th class="cell">${i}</th>`;
    html+='</tr></thead><tbody>';
    // decoding (argmax) row
    html+='<tr class="decode-row"><td><b>decoding</b> <span class="muted">argmax</span></td>';
    for(let i=0;i<t.n_instances;i++){ const p=t.probabilities[i][argmaxIdx(t.probabilities[i])];
      html+=`<td class="cell"><span class="tag dec">${esc(decoded[i])}</span><div class="muted mono" style="font-size:11px">${p.toFixed(2)}</div></td>`; }
    html+='</tr>';
    // one row per sampled decoding
    for(const s of samples){
      const labs=s.assignment_labels[t.concept]||[];
      const good=s.reward>0.5?'good':'';
      html+=`<tr class="${good}"><td>sample ${s.index} <span class="reward-${s.reward>0.5?1:0} mono">r=${s.reward.toFixed(2)}</span></td>`;
      for(let i=0;i<t.n_instances;i++){ const l=labs[i]; const match=(l===decoded[i]);
        html+=`<td class="cell"><span class="tag ${match?'match':''}">${esc(l)}${match?' ✓':''}</span></td>`; }
      html+='</tr>';
    }
    html+='</tbody></table></div></div>';
  }
  // outcome of each sampled decoding (generated sample + logp + reward)
  html+='<div style="margin:10px 0 6px"><b>Sample outcomes</b> <span class="colhint">— decoding → generated sample → reward</span></div>';
  html+='<div class="scroll" style="max-height:260px"><table><thead><tr><th>sample</th>'
      +'<th title="decoder(sampled decoding) — the input handed to the reward function">generated sample</th>'
      +'<th title="log p = sum over instances of log softmax(logits)[sampled class] — joint log-probability of this decoding under the model">log p</th>'
      +'<th title="reward_fn(generated sample) [+ constraint reward], reduced to a scalar">reward</th>'
      +'</tr></thead><tbody>';
  for(const s of samples){
    let go=s.generator_output; if(go===null||go===undefined)go='—'; else if(typeof go==='object')go=JSON.stringify(go);
    const rc=s.reward>0.5?'reward-1':'reward-0';
    html+=`<tr class="${s.reward>0.5?'good':''}"><td class="muted">${s.index}</td><td class="mono">${esc(go)}</td><td class="mono muted">${s.logprob.toFixed(3)}</td><td class="mono ${rc}">${s.reward.toFixed(3)}</td></tr>`;
  }
  html+='</tbody></table></div>';
  $('#samples').innerHTML=html;
}

function renderMetrics(st){
  const rs=st.reward_sources||{}; const src=[rs.function?'reward fn':null,rs.constraints?'constraints':null].filter(Boolean).join(' + ')||'—';
  const m=[
    ['step', st.step],['epoch', st.epoch??'—'],['estimator', st.estimator,'small'],
    ['samples', st.num_samples],['loss', (st.loss).toFixed(5),'loss'],
    ['mean reward', (st.mean_reward).toFixed(4),'reward'],['reward source', src,'small'],
  ];
  $('#metrics').innerHTML=m.map(([k,v,cls])=>`<div class="metric"><div class="k">${k}</div><div class="v ${cls||''}">${esc(v)}</div></div>`).join('');

  const rsrc=[];
  if(rs.function) rsrc.push('reward_fn(generated sample)');
  if(rs.constraints) rsrc.push('constraint satisfaction');
  const lossF = st.estimator==='reinforce'
    ? 'loss = − mean( (<span class="r">reward</span> − mean_reward) · <b>log p</b> )'
    : 'loss = −( logsumexp(<b>log p</b> + log <span class="r">reward</span>) − logsumexp(<b>log p</b>) )';
  $('#formula').innerHTML =
    `<b>log p</b> (per sampled decoding) = Σ over instances  log softmax(logits)[sampled class]   — joint log-probability of the decoding under the model\n`
    + `<span class="r">reward</span> (per sampled decoding) = ${esc(rsrc.join('  +  ')||'—')}\n`
    + `${lossF}   — estimator: ${esc(st.estimator)}`;
}

let lastKey=null;
async function refresh(){
  if(busy)return; busy=true;
  try{
    const r=await fetch('/api/state'); const d=await r.json();
    // Lightweight, every poll: status pill + buttons (does not touch the tables).
    const sp=$('#status'); sp.className='pill '+d.status; sp.textContent=d.status+(d.mode?(' · '+d.mode):'');
    $('#stepinfo').textContent=d.step_count?`${d.step_count} step(s) seen`:'';
    const waiting=d.status==='waiting'; const ended=(d.status==='stopped'||d.status==='done');
    $('#btnNext').disabled=ended||!(waiting||d.status==='running');
    $('#btnPlay').disabled=ended; $('#btnPause').disabled=ended; $('#btnStop').disabled=ended;
    const st=d.state;
    // Heavy tables: only rebuild when the step actually changes, otherwise the
    // 400ms poll would re-create the DOM each tick and reset the scrollbars.
    const key=st?(st.step+'|'+st.num_samples):'none';
    if(st && key!==lastKey){
      lastKey=key;
      renderMetrics(st); renderTargets(st.targets); renderSamples(st.samples, st.targets);
      const di=st.data_item||{}; const keys=Object.keys(di);
      if(keys.length){ $('#disec').style.display=''; $('#di').innerHTML=keys.map(k=>`<span class="tag"><b>${esc(k)}</b>: ${esc(typeof di[k]==='object'?JSON.stringify(di[k]):di[k])}</span>`).join(''); }
      else { $('#disec').style.display='none'; }
    }
  }catch(e){}
  busy=false;
}
setInterval(refresh,400); refresh();
</script>
</body>
</html>"""
