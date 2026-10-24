from __future__ import annotations

import json
import threading
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en-US"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EvoTailor · Live experiment dashboard</title>
<style>
:root{--bg:#081018;--panel:#101b27;--panel2:#142333;--line:#243649;--text:#e6eef7;--muted:#8ca0b5;--cyan:#49d6ca;--blue:#6ea8fe;--violet:#ac8cff;--green:#54d68b;--yellow:#f6c85f;--red:#ff7185;--shadow:0 20px 55px rgba(0,0,0,.28)}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -5%,#163752 0,transparent 32%),radial-gradient(circle at 95% 5%,#2b1e4b 0,transparent 28%),var(--bg);color:var(--text);font-family:Inter,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;min-height:100vh}
.shell{max-width:1560px;margin:auto;padding:26px}.top{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}.brand{display:flex;align-items:center;gap:13px}.logo{width:46px;height:46px;border-radius:14px;background:linear-gradient(135deg,var(--cyan),var(--violet));display:grid;place-items:center;font-weight:900;color:#081018;box-shadow:0 10px 28px rgba(73,214,202,.25)}h1{font-size:21px;margin:0;letter-spacing:.2px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.live{display:flex;align-items:center;gap:9px;padding:9px 13px;border:1px solid var(--line);border-radius:999px;background:rgba(16,27,39,.8);font-size:12px}.dot{width:9px;height:9px;border-radius:50%;background:var(--yellow);box-shadow:0 0 14px currentColor}.dot.ok{background:var(--green)}.dot.bad{background:var(--red)}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:13px;margin-bottom:15px}.card,.panel{background:linear-gradient(145deg,rgba(20,35,51,.94),rgba(13,23,34,.94));border:1px solid rgba(110,168,254,.15);border-radius:16px;box-shadow:var(--shadow)}.card{padding:16px}.label{font-size:11px;color:var(--muted);letter-spacing:.8px;margin-bottom:7px}.value{font-size:19px;font-weight:720;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.value.accent{color:var(--cyan)}
.replay-only{display:none}.replay-only.active{display:block}.replay-only input{width:100%}.progress{height:7px;border-radius:20px;background:#263545;overflow:hidden;margin-top:10px}.progress i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:.4s}
.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(360px,.65fr);gap:15px}.panel{padding:18px;min-width:0}.panel-title{font-weight:700;font-size:14px;margin-bottom:14px;display:flex;justify-content:space-between;align-items:center}.hint{font-weight:400;font-size:11px;color:var(--muted)}
.candidates{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.candidate{padding:13px;border:1px solid var(--line);border-radius:12px;background:rgba(8,16,24,.48)}.candidate.chosen{border-color:var(--cyan);box-shadow:inset 0 0 0 1px rgba(73,214,202,.22)}.cand-head{display:flex;justify-content:space-between;gap:8px;font-size:12px;font-weight:700}.pill{font-size:10px;color:var(--muted);padding:3px 7px;border-radius:999px;background:#1c2b3a}.params{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;margin-top:9px;color:var(--muted);font-size:10px}.params span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.prediction{margin-top:9px;color:var(--blue);font-size:11px}
.decision{border-left:3px solid var(--cyan);padding:13px 14px;background:rgba(73,214,202,.055);border-radius:5px 12px 12px 5px;line-height:1.65;font-size:13px}.decision strong{color:var(--cyan)}.risks{margin:10px 0 0;padding-left:18px;color:var(--muted);font-size:11px;line-height:1.6}.error{border-left-color:var(--red);background:rgba(255,113,133,.06)}
.results{display:flex;align-items:flex-end;gap:7px;height:120px;margin-top:8px;padding:6px 3px 22px}.bar{flex:1;min-width:15px;background:linear-gradient(180deg,var(--blue),#3e639d);border-radius:6px 6px 2px 2px;position:relative}.bar.best{background:linear-gradient(180deg,var(--green),#258054)}.bar label{position:absolute;top:calc(100% + 5px);left:50%;transform:translateX(-50%);font-size:9px;color:var(--muted)}
.timeline{max-height:540px;overflow:auto;padding-right:5px}.event{padding:11px 12px;border-left:2px solid var(--line);margin-left:5px;position:relative}.event:before{content:"";position:absolute;width:8px;height:8px;border-radius:50%;background:var(--blue);left:-5px;top:16px}.event-time{font-size:10px;color:var(--muted)}.event-step{font-size:12px;font-weight:700;margin:3px 0}.event-msg{font-size:11px;color:#b9c8d7;line-height:1.55}.empty{padding:28px;text-align:center;color:var(--muted);font-size:12px}.footer{margin-top:14px;color:var(--muted);font-size:10px;text-align:right}
@media(max-width:1050px){.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.candidates{grid-template-columns:1fr}}@media(max-width:600px){.shell{padding:14px}.top{align-items:flex-start}.stats{grid-template-columns:1fr 1fr}.candidates{grid-template-columns:1fr}}
</style></head><body><div class="shell">
<div class="top"><div class="brand"><div class="logo">EV</div><div><h1>EvoTailor Live experiment dashboard</h1><div class="sub">Candidate generation · Performance prediction · Agent decision · Trial feedback</div></div></div><div class="live"><span id="dot" class="dot"></span><span id="connection">Connecting to the dashboard</span></div></div>
<div class="stats">
 <div class="card replay-only" id="replay-card"><div class="label">Replay progress</div><div class="value accent" id="replay-label">Live</div><input id="replay-slider" type="range" min="0" max="0" value="0" step="1" disabled></div>
 <div class="card"><div class="label">Task</div><div class="value" id="campaign">—</div></div>
 <div class="card"><div class="label">Current trial</div><div class="value accent" id="trial">—</div><div class="progress"><i id="progress"></i></div></div>
 <div class="card"><div class="label">Episode progress</div><div class="value" id="episode">—</div></div>
 <div class="card"><div class="label">Trials in episode</div><div class="value" id="episode-trial">—</div></div>
 <div class="card"><div class="label">Total budget</div><div class="value" id="budget">—</div></div>
 <div class="card"><div class="label">Search phase</div><div class="value" id="phase">—</div></div>
 <div class="card"><div class="label">Best objective</div><div class="value" id="best">—</div></div>
 <div class="card"><div class="label">Current step</div><div class="value" id="step">Waiting for launch</div></div>
</div>
<div class="grid">
 <div><section class="panel"><div class="panel-title"><span>Project inspection and search-space initialization</span><span class="hint">Runs only before Trial 0</span></div><div id="project-info" class="decision">Waiting for project inspection</div></section>
 <section class="panel" style="margin-top:15px"><div class="panel-title"><span>Episode phase plan</span><span class="hint">Allocated by the agent within the budget</span></div><div id="episode-plan" class="empty">Waiting for episode planning</div></section>
 <section class="panel" style="margin-top:15px"><div class="panel-title"><span>Candidate comparison</span><span class="hint" id="prediction-mode">Waiting for candidates</span></div><div id="candidates" class="candidates"><div class="empty">Candidates will appear here after generation</div></div></section>
 <section class="panel" style="margin-top:15px"><div class="panel-title"><span>Agent decision</span><span class="hint" id="agent-model">—</span></div><div id="decision" class="empty">Waiting for the agent to compare candidates</div></section>
 <section class="panel" style="margin-top:15px"><div class="panel-title"><span>Trial results</span><span class="hint">Lower objective values are better</span></div><div id="results" class="results"><div class="empty">Waiting for trial results</div></div></section></div>
 <div><section class="panel"><div class="panel-title"><span>Episode review and policy evolution</span><span class="hint">Updated at the end of each episode</span></div><div id="evolution" class="empty">Waiting for the first episode review</div></section>
 <section class="panel" style="margin-top:15px"><div class="panel-title"><span>Live execution trace</span><span class="hint">Oldest first, newest last; refreshes every second</span></div><div id="timeline" class="timeline"><div class="empty">Waiting for run events</div></div></section></div>
</div><div class="footer">Available while this EvoTailor process runs; backed by persistent trace.jsonl</div></div>
<script>
const $=id=>document.getElementById(id);const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let lastCount=-1;let lastRenderKey="";let replayMode=false;let replayIndex=-1;let replayEvents=[];let replayFrames=[];
function detail(e,key,def=null){return e?.details?.[key]??def}function latest(events,pred){for(let i=events.length-1;i>=0;i--)if(pred(events[i]))return events[i];return null}
function parseTrial(msg){let m=String(msg).match(/trial\s+(\d+)\/(\d+)/i);return m?{now:+m[1],total:+m[2]}:null}
function buildFrames(events){const keep=new Set(['Task launch','Project inspection','Project features','Search space construction','Episode plan','New round','Step 1/6: candidates generated','Step 2/6: performance prediction','Step 3/6: agent verdict','Step 4/6: test result','Step 5/6: calibration feedback','Step 5/6: update history','Episode boundary decision','Step 6/6: reflection complete','Policy evolution','Policy unchanged','Policy evolution failed','Task finished']);const frames=[];events.forEach((e,i)=>{if(keep.has(e.step))frames.push(i)});return frames.length?frames:[Math.max(0,events.length-1)]}
function render(data){replayEvents=data.events||[];replayFrames=buildFrames(replayEvents);const rawIndex=replayMode&&replayIndex>=0?(replayFrames[replayIndex]??replayFrames.at(-1)):replayEvents.length-1;const ev=replayMode&&replayIndex>=0?replayEvents.slice(0,rawIndex+1):replayEvents,timelineRoot=$('timeline'),stickToBottom=lastCount<0||timelineRoot.scrollHeight-timelineRoot.scrollTop-timelineRoot.clientHeight<90;const renderKey=`${replayMode?replayIndex:-1}:${ev.length}:${ev.at(-1)?.timestamp||""}`;if(renderKey===lastRenderKey)return;lastRenderKey=renderKey;lastCount=ev.length;
 const start=latest(ev,e=>e.step==='Task launch'),round=latest(ev,e=>e.step==='New round'),decision=latest(ev,e=>e.step === 'Step 3/6: agent verdict'),pred=latest(ev,e=>e.step === 'Step 2/6: performance prediction'),end=latest(ev,e=>e.step==='Task finished'),project=latest(ev,e=>e.step==='Project features'||e.step==='Project inspection'),space=latest(ev,e=>e.step==='Search space construction'),plan=latest(ev,e=>e.step==='Episode plan'),phase=latest(ev,e=>e.step==='Trial phase decision'||e.step==='Episode plan'),evolutions=ev.filter(e=>['Policy evolution','Policy unchanged','Policy evolution failed'].includes(e.step));
 const tr=parseTrial(round?.message),resultEvents=ev.filter(e=>e.step === 'Step 4/6: test result'),episode=detail(round,'current_episode','—'),episodeTrial=detail(round,'episode_trial','—'),boundaryEvents=ev.filter(e=>e.step==='Episode boundary decision'&&detail(e,'decision')==='end_episode'),episodeBudgets=boundaryEvents.map(e=>Number(detail(e,'episode_trials'))).filter(Number.isFinite),startupError=latest(ev,e=>e.step==='Task launch failed');$('campaign').textContent=detail(start,'task_id','—');$('trial').textContent=tr?`${tr.now} / ${tr.total}`:'—';$('progress').style.width=tr?`${100*tr.now/tr.total}%`:'0%';$('episode').textContent=episode;$('episode-trial').textContent=episodeTrial;const totalBudget=tr?.total||detail(start,'total_rounds','—');$('budget').textContent=episodeBudgets.length?`${episodeBudgets.join(' + ')} = ${totalBudget} Trial`:`Dynamic episodes; total budget ${totalBudget} Trial`;$('phase').textContent=phase?.details?.current_stage||(round?.message.match(/phase: ([^.]+)\./)||[])[1]||'—';$('step').textContent=startupError?'Launch failed':(ev.at(-1)?.step||'Waiting for launch');$('agent-model').textContent=detail(start,'backing_model','—');
 if(startupError){$('project-info').className='decision error';$('project-info').innerHTML=`<strong>Task launch failed</strong><br>${esc(startupError.message)}<pre style="white-space:pre-wrap;color:var(--muted);font-size:11px">${esc(JSON.stringify(startupError.details||{},null,2))}</pre>`} else if(project||space){const rawSummaries=detail(project,'agent_summary_natural_language',[]),summaries=Array.isArray(rawSummaries)?rawSummaries:[rawSummaries],rawAdded=detail(space,'added_dimensions',[]),added=Array.isArray(rawAdded)?rawAdded:[rawAdded];$('project-info').innerHTML=`<strong>${esc(project?.message||'Project inspection status unknown')}</strong><ul class="risks">${summaries.filter(Boolean).map(x=>`<li>${esc(x)}</li>`).join('')}</ul><div>Search space: ${esc(detail(space,'base_dimension_count','—'))} → ${esc(detail(space,'final_dimension_count','—'))} dimensions; added: ${added.length?added.map(esc).join(', '):'none'}</div>`}
 if(plan){const phases=detail(plan,'stage_plan',[]),counts=detail(plan,'trials_per_stage',{});$('episode-plan').className='decision';$('episode-plan').innerHTML=`<strong>Episode ${esc(detail(plan,'Episode','—'))}</strong><br>${phases.map((x,i)=>`${i+1}. ${esc(x)}`).join(' ')}<br><span style="color:var(--muted)">${Object.entries(counts).map(([k,v])=>`${esc(k)} ${v}`).join(' · ')}</span><br>${esc(detail(plan,'planning_rationale',''))}`}
 if(evolutions.length){$('evolution').className='';$('evolution').innerHTML=evolutions.map(e=>`<div class="decision ${e.step.includes('failed')?'error':''}" style="margin-bottom:10px"><strong>${esc(e.message)}</strong><br>${esc(detail(e,'Reason',''))}<pre style="white-space:pre-wrap;color:var(--muted);font-size:10px">${esc(JSON.stringify(detail(e,'Change',detail(e,'current_policy',{})),null,2))}</pre></div>`).join('')}
 const vals=resultEvents.map(e=>Number(detail(e,'target_value'))).filter(Number.isFinite);$('best').textContent=vals.length?Math.min(...vals).toFixed(4):'—';
 const roundIndex=round?ev.indexOf(round):0,current=ev.slice(roundIndex),cand=current.filter(e=>e.step === 'Step 1/6: candidates generated'),preds=detail(pred,'predicted_result',[])||[],chosen=detail(decision,'final_parameters');$('prediction-mode').textContent=detail(pred,'prediction_method','Waiting for prediction');
 $('candidates').innerHTML=cand.length?cand.map(e=>{const p=detail(e,'candidate_parameters',{}),id=(e.message.match(/proposed candidate\s+([^\s.]+)\./)||[])[1]||'',source=e.message.split(' proposed candidate')[0],pr=preds.find(x=>x['candidate_id']===id),isChosen=chosen&&JSON.stringify(chosen)===JSON.stringify(p);return `<div class="candidate ${isChosen?'chosen':''}"><div class="cand-head"><span>${esc(source)}</span><span class="pill">${esc(id)}</span></div><div class="params">${Object.entries(p).map(([k,v])=>`<span title="${esc(k)}">${esc(k.split('.').at(-1))}: ${esc(v)}</span>`).join('')}</div>${pr?`<div class="prediction">Calibrated prediction ${Number(pr['Calibrated prediction']).toFixed(3)} · Uncertainty ${Number(pr['Uncertainty']).toFixed(3)}</div>`:''}</div>`}).join(''):'<div class="empty">Candidates will appear here after generation</div>';
 if(decision){const fallback=detail(decision,'fallback_used',false),risks=detail(decision,'risk_factors',[]);$('decision').className=`decision ${fallback?'error':''}`;$('decision').innerHTML=`<strong>${esc(decision.message)}</strong><br>${esc(detail(decision,'decision_rationale',''))}${fallback?'<br><b style="color:var(--red)">This trial did not use a model decision</b>':''}<ul class="risks">${risks.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`}
 const max=Math.max(...vals,1),min=Math.min(...vals,0),span=Math.max(max-min,.001),best=Math.min(...vals);$('results').innerHTML=vals.length?vals.map((v,i)=>`<div class="bar ${v===best?'best':''}" style="height:${28+72*(max-v)/span}%" title="Trial ${i+1}: ${v}"><label>${i+1}</label></div>`).join(''):'<div class="empty">Waiting for trial results</div>';
 timelineRoot.innerHTML=ev.length?ev.slice(-80).map(e=>`<article class="event"><div class="event-time">${new Date(e.timestamp).toLocaleTimeString('en-US',{hour12:false})}</div><div class="event-step">${esc(e.step)}</div><div class="event-msg">${esc(e.message)}</div></article>`).join(''):(startupError?'<div class="decision error">No trial has started. See the launch failure above.</div>':'<div class="empty">Waiting for run events</div>');if(stickToBottom)timelineRoot.scrollTop=timelineRoot.scrollHeight;
 if(end){$('connection').textContent='Task completed';$('dot').className='dot ok'} }
function setupReplay() {
 const card=$('replay-card'), slider=$('replay-slider');
 if(!replayMode)return;
 card.classList.add('active'); slider.disabled=false;
 slider.max=Math.max(0,replayFrames.length-1);
 slider.value=replayIndex;
 $('replay-label').textContent=`${replayIndex+1} / ${replayFrames.length}`;
 if(slider.dataset.bound!=='1') {
  slider.dataset.bound='1';
  slider.oninput=()=>{
   replayIndex=Number(slider.value); lastCount=-1; lastRenderKey='';
   $('replay-label').textContent=`${replayIndex+1} / ${replayFrames.length}`;
   render({events:replayEvents});
  };
 }
}
async function poll() {
 try {
  const r=await fetch('/api/state?ts='+Date.now(),{cache:'no-store'});
  if(!r.ok)throw Error(`HTTP ${r.status}`);
  const data=await r.json(); replayMode=Boolean(data.replay);
  if(replayMode) {
   replayEvents=data.events||[]; replayFrames=buildFrames(replayEvents);
   if(replayIndex<0)replayIndex=Math.max(0,replayFrames.length-1);
   replayIndex=Math.min(replayIndex,replayFrames.length-1);
   setupReplay();
  }
  render(data);
  $('connection').textContent=replayMode?`Replay mode · ${replayEvents.length} events`:'Live connection ready';
  $('dot').className='dot ok';
 } catch(e) {
  $('connection').textContent=`Connection failed: ${e.message}; retrying`;
  $('dot').className='dot bad';
 }
}
poll(); setInterval(poll,1000);
</script></body></html>"""


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class DashboardServer:
    def __init__(
        self,
        run_dir: Path,
        *,
        host: str = "0.0.0.0",
        port: int = 8090,
        replay: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.host = host
        self.port = port
        self.replay = replay
        self._server: _ReusableThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _state(self) -> dict[str, Any]:
        events = []
        path = self.run_dir / "trace.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[-2000:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        summary = None
        summary_path = self.run_dir / "summary.json"
        if summary_path.exists():
            with suppress(json.JSONDecodeError):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {"events": events, "summary": summary, "replay": self.replay}

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                route = self.path.split("?", 1)[0]
                if route == "/":
                    self._send(DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
                elif route == "/api/state":
                    self._send(
                        json.dumps(owner._state(), ensure_ascii=False).encode(),
                        "application/json; charset=utf-8",
                    )
                elif route == "/api/health":
                    self._send(b'{"ok":true}', "application/json")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def _send(self, body: bytes, content_type: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        try:
            self._server = _ReusableThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot start dashboard at {self.host}:{self.port}; the port may be in use: {exc}"
            ) from exc
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="evotailor-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._server = None
        self._thread = None
