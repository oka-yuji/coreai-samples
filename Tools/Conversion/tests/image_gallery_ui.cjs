const fs = require("fs");
const vm = require("vm");
const template=fs.readFileSync(process.argv[2],"utf8");
const html=fs.readFileSync(process.argv[3],"utf8");
const payload=JSON.parse(html.match(/<script id="payload" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const source=template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function element(tag){const queried=new Map();return {tag,children:[],attributes:{},textContent:"",value:"",files:[],append(...v){this.children.push(...v)},replaceChildren(...v){this.children=v},setAttribute(k,v){this.attributes[k]=v},querySelector(key){if(!queried.has(key))queried.set(key,element(key));return queried.get(key)},scrollIntoView(options){this.scrollOptions=options},click(){}};}
function make(stored){
 const elements=new Map();
 const get=id=>{if(!elements.has(id))elements.set(id,element(id));return elements.get(id)};
 get("payload").textContent=JSON.stringify(payload);
 const saved=new Map(stored?[["qmi-review-"+payload.comparison_sha256,stored]]:[]);
 const context=vm.createContext({document:{getElementById:get,createElement:element},structuredClone,localStorage:{getItem:k=>saved.get(k)||null,setItem:(k,v)=>saved.set(k,v)},Blob,URL,setTimeout});
 vm.runInContext(source,context);
 return {get,context,saved,score:()=>vm.runInContext("ratings.cohorts.C.ratings[0].axes.instruction",context)};
}
function scores(score){const value=structuredClone(payload.ratings);value.cohorts.C.ratings[0].axes.instruction=score;return JSON.stringify(value);}
function controls(root){return [root,...root.children.flatMap(controls)].filter(value=>value.tag==="select");}
async function latestSelection(){
 const state=make();let oldDone,newDone;
 state.get("resume").files=[{text:()=>new Promise(r=>oldDone=r)}];state.get("resume").value="old.json";
 const earlier=state.get("resume").onchange();
 state.get("resume").files=[{text:()=>new Promise(r=>newDone=r)}];state.get("resume").value="new.json";
 const newer=state.get("resume").onchange();
 newDone(scores(2));await newer;const afterLatest=state.score();
 oldDone(scores(0));await earlier;
 return {case:"last_selected_file_wins",afterLatest,afterStale:state.score(),passed:state.score()===2};
}
async function editDuringResume(){
 const state=make();let done;
 state.get("resume").files=[{text:()=>new Promise(r=>done=r)}];state.get("resume").value="old.json";
 const restore=state.get("resume").onchange();
 const control=controls(state.get("columns")).find(e=>e.attributes["data-cohort"]==="C"&&e.attributes["data-axis"]==="instruction");
 control.value="2";control.onchange();const afterEdit=state.score();
 done(scores(0));await restore;
 return {case:"manual_edit_invalidates_pending_restore",afterEdit,afterStale:state.score(),inputValue:state.get("resume").value,passed:state.score()===2};
}
function restoredAfterDisplayChange(){
 const stored=scores(1),state=make(stored);
 const control=controls(state.get("columns")).find(e=>e.attributes["data-cohort"]==="C"&&e.attributes["data-axis"]==="instruction");
 return {case:"existing_scores_survive_japanese_display",passed:state.score()===1&&control.value==="1"&&state.saved.get("qmi-review-"+payload.comparison_sha256)===stored};
}
function caseContext(){
 const state=make();let passed=true;
 for(const [index,item] of payload.cases.entries()){
  state.get("case").value=String(index);state.get("case").onchange();
  passed=passed&&state.get("prompt").textContent===item.guidance.prompt_ja&&state.get("prompt-source").textContent===item.prompt;
  passed=passed&&JSON.stringify(state.get("instruction-checks").children.map(x=>x.textContent))===JSON.stringify(item.guidance.checks.instruction);
  passed=passed&&state.get("reference").hidden===!item.reference;
  if(item.reference)passed=passed&&state.get("reference").querySelector("img").src===item.reference;
  for(const cohort of ["A0","A1","B","C"]){
   const selected=controls(state.get("columns")).filter(e=>e.attributes["data-cohort"]===cohort);
   passed=passed&&selected[0].attributes["data-axis"]==="instruction"&&selected.length===Object.keys(item.guidance.checks).length;
  }
 }
 return {case:"all_sixty_contexts_match_bound_prompts_and_references",passed};
}
function navigationDoesNotRate(){
 const state=make(),before=vm.runInContext("JSON.stringify(ratings)",state.context);
 for(let i=0;i<59;i++)state.get("next").onclick();
 state.get("previous").onclick();
 return {case:"navigation_never_creates_ratings",passed:before===vm.runInContext("JSON.stringify(ratings)",state.context)&&state.saved.size===0};
}
(async()=>{const results=[await latestSelection(),await editDuringResume(),restoredAfterDisplayChange(),caseContext(),navigationDoesNotRate()];console.log(JSON.stringify({node:process.version,results},null,2));process.exitCode=results.every(row=>row.passed)?0:1;})().catch(error=>{console.error(error);process.exitCode=2});
