"""Self-contained HTML notes table for manual tracking-gap entry reviews."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shlex


WORKSHEET_JS = r'''
const annotationFields = ['review_id','review_status','observed_entry_frames','confidence','reviewer','notes'];
function validationError(row, episode) {
  const status = row.review_status;
  if (!['pending','reviewed','unscorable'].includes(status)) return 'Choose a valid review status.';
  if (status !== 'reviewed') return row.observed_entry_frames.trim() ? 'Only reviewed cases may have an entry list; put partial observations in notes.' : '';
  let frames;
  try { frames = JSON.parse(row.observed_entry_frames); }
  catch (_) { return 'Enter a JSON frame list, such as [49262], or [] for zero entries.'; }
  if (!Array.isArray(frames) || !frames.every(Number.isInteger)) return 'Entry frames must be a list of integers.';
  if (new Set(frames).size !== frames.length) return 'Remove duplicate entry frames.';
  if (frames.some(f => f < Number(episode.review_start_frame) || f >= Number(episode.review_stop_frame_exclusive))) return 'An entry frame is outside the scored interval.';
  if (!['high','medium','low'].includes(row.confidence)) return 'Choose a confidence for a reviewed case.';
  return '';
}
function annotationsCSV(rows) {
  const quote = value => '"' + String(value ?? '').replace(/"/g,'""') + '"';
  return annotationFields.map(quote).join(',') + '\r\n' +
    rows.map(row => annotationFields.map(k => quote(row[k])).join(',')).join('\r\n');
}
function caseNumber(episode, index) {
  return episode.case_number || index + 1;
}
if (typeof document !== 'undefined') {
  const data = JSON.parse(document.getElementById('review-data').textContent);
  const tableRows = Array.from(document.querySelectorAll('tbody tr[data-review-id]'));
  const episodeById = new Map(data.episodes.map(ep => [ep.review_id, ep]));
  const message = document.getElementById('message');
  const key = 'tracking-gap-notes:' + location.href + ':' + data.episodes.map(e => e.review_id).join(',') + ':' + JSON.stringify(data.annotations);
  function readRows() {
    return tableRows.map(tr => {
      const row = {review_id:tr.dataset.reviewId};
      tr.querySelectorAll('[data-field]').forEach(input => row[input.dataset.field] = input.value);
      return row;
    });
  }
  function fillRows(rows) {
    const index = new Map(rows.map(row => [row.review_id,row]));
    tableRows.forEach(tr => {
      const row = index.get(tr.dataset.reviewId);
      if (row) tr.querySelectorAll('[data-field]').forEach(input => input.value = row[input.dataset.field] ?? '');
    });
  }
  function refresh() {
    let reviewed = 0, unscorable = 0;
    tableRows.forEach(tr => {
      const status = tr.querySelector('[data-field="review_status"]').value;
      reviewed += status === 'reviewed'; unscorable += status === 'unscorable';
      tr.dataset.status = status;
      const query = document.getElementById('filter').value.toLowerCase();
      const circle = document.getElementById('circle-filter').value;
      const statusFilter = document.getElementById('status-filter').value;
      tr.hidden = !(tr.dataset.search.includes(query) && (!circle || tr.dataset.circle === circle) && (!statusFilter || status === statusFilter));
    });
    document.getElementById('progress').textContent = reviewed + ' reviewed · ' + unscorable + ' unscorable · ' + (tableRows.length-reviewed-unscorable) + ' pending';
  }
  function saveDraft() {
    try { localStorage.setItem(key, JSON.stringify(readRows())); message.textContent = 'Browser draft saved. Download CSV or HTML to keep a portable copy.'; }
    catch (_) { message.textContent = 'Browser autosave unavailable. Download an HTML draft regularly.'; }
    refresh();
  }
  function download(name, contents, mime) {
    const url = URL.createObjectURL(new Blob([contents], {type:mime}));
    const link = document.createElement('a'); link.href = url; link.download = name;
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  fillRows(data.annotations);
  try {
    const saved = JSON.parse(localStorage.getItem(key) || 'null');
    if (Array.isArray(saved) && saved.length === tableRows.length && new Set(saved.map(r=>r.review_id)).size === tableRows.length && saved.every(r=>episodeById.has(r.review_id))) {
      fillRows(saved); message.textContent = 'Restored the browser draft. Download CSV or HTML to keep a portable copy.';
    }
  } catch (_) { message.textContent = 'Browser autosave unavailable. Download an HTML draft regularly.'; }
  tableRows.forEach(tr => {
    tr.addEventListener('input', saveDraft); tr.addEventListener('change', saveDraft);
    tr.querySelector('button[data-command]').addEventListener('click', async event => {
      const command = event.currentTarget.dataset.command;
      try { await navigator.clipboard.writeText(command); message.textContent = 'Player command copied.'; }
      catch (_) { window.prompt('Copy this command:', command); }
    });
  });
  ['filter','circle-filter','status-filter'].forEach(id => document.getElementById(id).addEventListener('input',refresh));
  document.getElementById('download-csv').addEventListener('click', () => {
    const rows = readRows(); tableRows.forEach(tr => tr.classList.remove('invalid'));
    for (let i=0; i<rows.length; i++) {
      const error = validationError(rows[i],episodeById.get(rows[i].review_id));
      if (error) {
        ['filter','circle-filter','status-filter'].forEach(id => document.getElementById(id).value = ''); refresh();
        tableRows[i].classList.add('invalid'); tableRows[i].scrollIntoView({block:'center'});
        message.textContent = 'Case ' + caseNumber(episodeById.get(rows[i].review_id), i) + ': ' + error + ' An HTML draft can still be downloaded.'; return;
      }
    }
    download('annotations.csv', annotationsCSV(rows), 'text/csv;charset=utf-8');
    message.textContent = 'Downloaded all cases, including filtered-out pending cases. Use this CSV with summarize --annotations.';
  });
  document.getElementById('download-html').addEventListener('click', () => {
    const clone = document.documentElement.cloneNode(true);
    clone.querySelector('#review-data').textContent = JSON.stringify({...data, annotations:readRows()}).replace(/</g,'\\u003c');
    clone.querySelector('#message').textContent = 'Portable HTML draft. Continue editing or download annotations.csv.';
    clone.querySelectorAll('tr[hidden]').forEach(tr => tr.removeAttribute('hidden'));
    download('circle_entry_notes.html', '<!doctype html>\n'+clone.outerHTML, 'text/html;charset=utf-8');
    message.textContent = 'Downloaded an editable HTML draft containing your current notes, including incomplete entries.';
  });
  refresh();
}
'''


def render_worksheet(episodes, annotations, *, bundle, python, script):
    ids = [ep["review_id"] for ep in episodes]
    annotation_ids = [row["review_id"] for row in annotations]
    if len(set(ids)) != len(ids) or len(set(annotation_ids)) != len(annotation_ids) or set(ids) != set(annotation_ids):
        raise ValueError("Worksheet annotations must match episode IDs exactly without duplicates")
    index = {r["review_id"]: r for r in annotations}
    esc = lambda value: html.escape(str(value), quote=True)

    def select(field, values, row):
        options = []
        for value, label in values:
            selected = ' selected' if row.get(field, "") == value else ''
            options.append(f'<option value="{esc(value)}"{selected}>{esc(label)}</option>')
        return f'<select data-field="{field}" aria-label="{field}">' + ''.join(options) + '</select>'

    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Manual circle-interaction notes</title><style>',
             'body{font-family:system-ui,sans-serif;margin:1.5rem;color:#222;line-height:1.45}h1{font-size:1.5rem}p{max-width:110ch}',
             '.toolbar{display:flex;gap:.7rem;flex-wrap:wrap;align-items:center;margin:1rem 0}.table-wrap{overflow:auto;max-height:75vh;border:1px solid #ccc}',
             'table{border-collapse:separate;border-spacing:0;font-size:.88rem;width:100%}th,td{padding:.55rem;border-right:1px solid #ddd;border-bottom:1px solid #ddd;vertical-align:top}',
             'th{position:sticky;top:0;background:#eaf0f5;z-index:1;text-align:left}td.video{min-width:240px}td.interval{min-width:155px}td.notes{min-width:270px}',
             'textarea,input,select,button{font:inherit;padding:.4rem}textarea{box-sizing:border-box;width:100%;min-width:150px;min-height:5rem}textarea[data-field="notes"]{min-height:7rem}input[data-field="reviewer"]{width:7rem}',
             'tr[data-status="reviewed"]{background:#f1f8f1}tr[data-status="unscorable"]{background:#fff7e9}tr.invalid{outline:3px solid #b33;outline-offset:-3px}',
             'small{display:block;color:#555}code{overflow-wrap:anywhere}details{margin:.4rem 0}#message{min-height:1.5em;color:#384b62}',
             '@media print{.toolbar{display:none}.table-wrap{max-height:none;overflow:visible}th{position:static}textarea{min-height:7rem}}',
             '</style></head><body><h1>Manual circle-interaction notes</h1>',
             '<p>One row per review case, in player order. Record entries into the <strong>named circle</strong> during the entire scored interval. Frames are zero-based; <strong>[start, stop) includes start and excludes stop</strong>. Context-only frames help interpretation but are not scored.</p>',
             '<p>For completed cases choose <strong>Reviewed</strong>, enter absolute entry frames as a JSON list (e.g. <code>[49262, 49270]</code>) or explicitly <code>[]</code> for zero, and choose confidence. Use <strong>Unscorable</strong> when visibility is insufficient and describe why in notes. Calculated events are deliberately omitted from this worksheet.</p>',
             '<p>Browser drafts are saved locally when available, not back into the source HTML or CSV. Download an HTML draft to retain the full table and notes, or download annotations.csv for the analysis pipeline. Downloads include all cases even when the table is filtered.</p>',
             '<div class="toolbar"><button id="download-csv">Download annotations.csv</button><button id="download-html">Download editable HTML draft</button><span id="progress"></span></div>',
             '<div class="toolbar"><label>Find <input id="filter" type="search" placeholder="Video, fly, role, or review ID"></label><label>Circle <select id="circle-filter"><option value="">All</option><option>reward</option><option>control</option></select></label>',
             '<label>Status <select id="status-filter"><option value="">All</option><option value="pending">Pending</option><option value="reviewed">Reviewed</option><option value="unscorable">Unscorable</option></select></label></div>',
             '<p id="message" role="status">Notes stay in this browser until you download a copy.</p>',
             '<div class="table-wrap"><table><thead><tr>',
             *[f'<th scope="col">{esc(label)}</th>' for label in (
                 '#', 'Video filename', 'Fly ID', 'Role', 'Circle', 'Scored interval [start, stop)',
                 'Training / sync buckets', 'Case type', 'Review status', 'Observed entry frames',
                 'Confidence', 'Observations / uncertainty', 'Reviewer')], '</tr></thead><tbody>']
    for i, ep in enumerate(episodes, 1):
        row = index[ep["review_id"]]
        start, stop = int(ep["review_start_frame"]), int(ep["review_stop_frame_exclusive"])
        command = shlex.join([str(python), str(script), 'play', '--bundle', str(bundle), '--review-id', ep["review_id"]])
        search = ' '.join(str(ep[k]) for k in ('video_path', 'fly', 'role', 'circle', 'review_id')).lower()
        duration = f'{float(ep["duration_s"]):.2f} s' if ep.get("duration_s") is not None else ''
        parts += [f'<tr data-review-id="{esc(ep["review_id"])}" data-circle="{esc(ep["circle"])}" data-search="{esc(search)}" data-status="{esc(row["review_status"])}">',
                  f'<td>{esc(ep.get("case_number", i))}<small><code>{esc(ep["review_id"])}</code></small></td>',
                  f'<td class="video">{esc(Path(ep["video_path"]).name)}<details><summary>Full path / player</summary><code>{esc(ep["video_path"])}</code><p><button type="button" data-command="{esc(command)}">Copy player command</button></p></details></td>',
                  f'<td>{esc(ep["fly"])}</td><td>{esc(ep["role"])}</td><td>{esc(ep["circle"])}</td>',
                  f'<td class="interval"><code>[{start}, {stop})</code><small>Inclusive last frame: {stop-1}</small><small>{esc(duration)}</small></td>',
                  f'<td>T{esc(ep.get("training", 2))}<small>SB: {esc(ep.get("sync_buckets") or "Outside full sync buckets")}</small></td>',
                  f'<td>{esc(ep.get("interval_type", ep.get("source", "gap_review")))}<small>{esc(ep.get("strata", ""))}</small><small>Selected gaps: {esc(ep.get("selected_gap_ids", ""))}</small></td>',
                  '<td>'+select('review_status', [('pending','Pending'),('reviewed','Reviewed'),('unscorable','Unscorable')], row)+'</td>',
                  f'<td><textarea data-field="observed_entry_frames" aria-label="Observed entry frames" placeholder="[49262] or []">{esc(row.get("observed_entry_frames", ""))}</textarea></td>',
                  '<td>'+select('confidence', [('','Choose…'),('high','High'),('medium','Medium'),('low','Low')], row)+'</td>',
                  f'<td class="notes"><textarea data-field="notes" aria-label="Observations and uncertainty" placeholder="Visible entries/exits; wall walking; occlusion; identity or boundary uncertainty">{esc(row.get("notes", ""))}</textarea></td>',
                  f'<td><input data-field="reviewer" aria-label="Reviewer" value="{esc(row.get("reviewer", ""))}"></td></tr>']
    payload = json.dumps(dict(episodes=episodes, annotations=annotations), ensure_ascii=True).replace('<', '\\u003c')
    parts += ['</tbody></table></div>',
              '<p>Entry convention: outside beyond the outer border followed by inside the nominal circle; border observations retain the last stable state. An animal already inside is not automatically an entry. Record exits and other interactions in notes when useful.</p>',
              f'<script type="application/json" id="review-data">{payload}</script>',
              '<script>'+WORKSHEET_JS+'</script></body></html>']
    return '\n'.join(parts)
