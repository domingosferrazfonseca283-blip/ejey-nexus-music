const $=id=>document.getElementById(id);
const LIBRARY_KEY="ejey-nexus-library-v2",ENDPOINT_KEY="ejey-nexus-endpoint-v1",TOKEN_KEY="ejey-nexus-token-v1";
let currentPrompt="";
let coverReady=false;
function readLibrary(){try{return JSON.parse(localStorage.getItem(LIBRARY_KEY)||"[]")}catch{return[]}}
function buildPrompt(){
 const idea=$("idea").value.trim();
 if(!idea){$("result").textContent="Adiciona uma direção criativa antes de construir o briefing.";return""}
 currentPrompt=[
  "EJEY NEXUS MUSIC — PRODUCTION BRIEF",
  "Create an original musical composition.",
  "Genre: "+$("genre").value+".",
  "Mood: "+$("mood").value+".",
  "Tempo: "+$("tempo").value+".",
  "Vocal arrangement: "+$("vocals").value+".",
  "Creative direction: "+idea,
  "Use original melodies, arrangements and performances. Do not imitate a specific living artist or use recordings without permission."
 ].join("\n");
 $("result").textContent=currentPrompt;
 return currentPrompt;
}
function renderLibrary(){
 const host=$("libraryList"),items=readLibrary();
 host.replaceChildren();
 if(!items.length){const p=document.createElement("p");p.style.cssText="color:var(--muted);font-size:12px";p.textContent="Ainda não existem projetos guardados.";host.append(p);return}
 items.forEach((item,index)=>{
  const row=document.createElement("div");row.className="track";
  const info=document.createElement("div");
  const title=document.createElement("strong");title.textContent=item.title||"Projeto sem título";
  const meta=document.createElement("span");meta.textContent=(item.genre||"Sem género")+" · "+(item.date||"");
  info.append(title,meta);
  const actions=document.createElement("div");actions.className="actions";actions.style.margin="0";
  const open=document.createElement("button");open.className="btn secondary";open.textContent="Abrir";open.onclick=()=>{ $("idea").value=item.idea||"";$("genre").value=item.genre||"Outro";currentPrompt=item.prompt||"";$("result").textContent=currentPrompt||"Briefing carregado.";if(item.coverTitle)$("coverTitle").value=item.coverTitle;if(item.artistName)$("artistName").value=item.artistName;if(item.coverStyle)$("coverStyle").value=item.coverStyle;if(item.coverPalette)$("coverPalette").value=item.coverPalette;if(item.coverData){const img=new Image();img.onload=()=>{const c=$("coverCanvas").getContext("2d");c.clearRect(0,0,1000,1000);c.drawImage(img,0,0,1000,1000);coverReady=true};img.src=item.coverData}}; 
  const del=document.createElement("button");del.className="btn ghost";del.textContent="Apagar";del.onclick=()=>{const next=readLibrary();next.splice(index,1);localStorage.setItem(LIBRARY_KEY,JSON.stringify(next));renderLibrary()};
  actions.append(open,del);row.append(info,actions);host.append(row);
 });
}
function baseUrl(){return $("endpoint").value.trim().replace(/\/+$/,"")}
async function api(path,options={}){
 const base=baseUrl();if(!base)throw Error("Configura primeiro a URL do servidor no computador.");
 const headers={...(options.headers||{})};if(options.body)headers["Content-Type"]="application/json";
 const token=$("apiToken").value.trim();if(token)headers.Authorization="Bearer "+token;
 const response=await fetch(base+path,{...options,headers,signal:AbortSignal.timeout(12000)});
 let data={};try{data=await response.json()}catch{data={detail:"Resposta inválida do servidor."}}
 return {response,data};
}
async function testConnection(){
 $("connect").disabled=true;$("connectionStatus").textContent="A testar ligação ao servidor...";
 try{const {response,data}=await api("/health");if(!response.ok)throw Error(data.detail||"Erro HTTP "+response.status);
 $("connectionStatus").textContent="Ligação estabelecida · API "+(data.api_version||"ativa")+" · Motor: "+(data.engine_status||"desconhecido")+". "+(data.message||"");
 }catch(e){$("connectionStatus").textContent="Não foi possível ligar: "+e.message}
 finally{$("connect").disabled=false}
}
async function sendGeneration(){
 const prompt=currentPrompt||buildPrompt();if(!prompt)return;
 $("generate").disabled=true;$("generationStatus").textContent="A enviar briefing ao servidor...";
 try{
  const {response,data}=await api("/v1/generate",{method:"POST",body:JSON.stringify({prompt,genre:$("genre").value,mood:$("mood").value,tempo:$("tempo").value,vocals:$("vocals").value})});
  $("generationStatus").textContent=data.message||data.detail||("Resposta HTTP "+response.status);
 }catch(e){$("generationStatus").textContent="Pedido não concluído: "+e.message}
 finally{$("generate").disabled=false}
}
function hashText(value){let h=2166136261;for(let i=0;i<value.length;i++){h^=value.charCodeAt(i);h=Math.imul(h,16777619)}return h>>>0}
function rng(seed){let s=seed>>>0;return()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296}}
const palettes={
 lime:{bg:"#10170f",a:"#c6f279",b:"#597d43",c:"#f0e9d8",d:"#222d20"},
 violet:{bg:"#100e1c",a:"#b9a1ff",b:"#65518f",c:"#f2eaff",d:"#27213b"},
 ocean:{bg:"#071820",a:"#71d7df",b:"#277b91",c:"#e1f7f2",d:"#102c3b"},
 ember:{bg:"#21140f",a:"#f0ad66",b:"#a64f32",c:"#f8e2c5",d:"#39251d"},
 mono:{bg:"#101010",a:"#f1f1eb",b:"#777777",c:"#ffffff",d:"#292929"}
};
function drawCover(){
 const canvas=$("coverCanvas"),ctx=canvas.getContext("2d"),title=$("coverTitle").value.trim()||"UNTITLED",artist=$("artistName").value.trim()||"EJEY NEXUS";
 const style=$("coverStyle").value,p=palettes[$("coverPalette").value]||palettes.lime,rand=rng(hashText(title+"|"+artist+"|"+style+"|"+$("coverPalette").value));
 ctx.clearRect(0,0,1000,1000);ctx.fillStyle=p.bg;ctx.fillRect(0,0,1000,1000);
 const bg=ctx.createLinearGradient(0,0,1000,1000);bg.addColorStop(0,p.d);bg.addColorStop(.55,p.bg);bg.addColorStop(1,"#050608");ctx.fillStyle=bg;ctx.fillRect(0,0,1000,1000);
 if(style==="minimal"){
  ctx.strokeStyle=p.b;ctx.lineWidth=2;for(let i=0;i<9;i++){ctx.beginPath();ctx.arc(500,450,95+i*43,0,Math.PI*2);ctx.stroke()}
  ctx.fillStyle=p.a;ctx.beginPath();ctx.arc(500,450,105,0,Math.PI*2);ctx.fill();
 }else if(style==="editorial"){
  ctx.fillStyle=p.a;ctx.fillRect(80,90,840,12);ctx.fillRect(80,115,270,4);
  ctx.save();ctx.translate(500,490);ctx.rotate(-.18);ctx.fillStyle=p.b;ctx.fillRect(-260,-260,520,520);ctx.fillStyle=p.a;ctx.fillRect(-210,-210,420,420);ctx.fillStyle=p.bg;ctx.fillRect(-155,-155,310,310);ctx.restore();
  ctx.strokeStyle=p.c;ctx.lineWidth=3;ctx.strokeRect(80,80,840,840);
 }else{
  const cx=style==="afrofuture"?560:500,cy=style==="cinematic"?420:490;
  for(let i=0;i<13;i++){
   const radius=90+i*31;ctx.beginPath();ctx.ellipse(cx,cy,radius,radius*(.42+rand()*.35),rand()*.8,0,Math.PI*2);ctx.strokeStyle=i%3===0?p.a:p.b;ctx.globalAlpha=.22+rand()*.45;ctx.lineWidth=2+rand()*8;ctx.stroke();
  }ctx.globalAlpha=1;
  for(let i=0;i<34;i++){
   const x=rand()*1000,y=rand()*760,r=2+rand()*13;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=i%4===0?p.c:p.a;ctx.globalAlpha=.15+rand()*.55;ctx.fill();
  }ctx.globalAlpha=1;
  if(style==="afrofuture"){
   ctx.strokeStyle=p.a;ctx.lineWidth=3;for(let i=0;i<7;i++){ctx.beginPath();ctx.moveTo(90+i*22,160);ctx.lineTo(500,270+i*19);ctx.lineTo(910-i*22,160);ctx.stroke()}
  }
 }
 const shade=ctx.createLinearGradient(0,590,0,1000);shade.addColorStop(0,"#00000000");shade.addColorStop(1,"#000000d9");ctx.fillStyle=shade;ctx.fillRect(0,540,1000,460);
 ctx.fillStyle=p.c;ctx.globalAlpha=.85;ctx.font="700 19px Arial";ctx.letterSpacing="5px";ctx.fillText("EJEY NEXUS   /   ORIGINAL RELEASE",82,90);ctx.globalAlpha=1;
 ctx.fillStyle=p.a;ctx.fillRect(82,770,54,5);
 const words=title.toUpperCase().split(/\s+/);let lines=[],line="";
 for(const word of words){const test=line?line+" "+word:word;ctx.font="900 76px Arial";if(ctx.measureText(test).width>820&&line){lines.push(line);line=word}else line=test}if(line)lines.push(line);
 const maxLines=3,shown=lines.slice(0,maxLines),fontSize=shown.length===1?88:shown.length===2?74:62;
 ctx.font="900 "+fontSize+"px Arial";ctx.fillStyle=p.c;ctx.textBaseline="alphabetic";
 shown.forEach((t,i)=>ctx.fillText(t,82,850+i*(fontSize+4),830));
 ctx.fillStyle=p.c;ctx.globalAlpha=.8;ctx.font="600 22px Arial";ctx.letterSpacing="3px";ctx.fillText(artist.toUpperCase(),84,950,820);
 ctx.globalAlpha=1;ctx.strokeStyle="#ffffff26";ctx.lineWidth=2;ctx.strokeRect(25,25,950,950);
 coverReady=true;$("coverStatus").textContent="Capa criada no dispositivo. Exporta em PNG ou guarda-a no projeto.";
}
function downloadCover(){
 if(!coverReady)drawCover();
 const link=document.createElement("a");link.download=(($("coverTitle").value.trim()||"ejey-nexus-cover").toLowerCase().replace(/[^a-z0-9]+/g,"-"))+".png";link.href=$("coverCanvas").toDataURL("image/png");link.click();
}
function saveProject(){
 const prompt=currentPrompt||buildPrompt();if(!prompt)return;
 const coverData=coverReady?$("coverCanvas").toDataURL("image/jpeg",.88):"";
 const items=readLibrary();items.unshift({
  title:$("coverTitle").value.trim()||$("idea").value.trim().slice(0,48)||"Novo projeto",
  idea:$("idea").value.trim(),genre:$("genre").value,prompt,
  date:new Date().toLocaleString("pt-PT"),
  coverTitle:$("coverTitle").value,artistName:$("artistName").value,
  coverStyle:$("coverStyle").value,coverPalette:$("coverPalette").value,coverData
 });
 try{localStorage.setItem(LIBRARY_KEY,JSON.stringify(items));renderLibrary();$("result").textContent="Projeto guardado na biblioteca local."}
 catch(e){$("result").textContent="Não foi possível guardar. Exporta a capa e liberta espaço no armazenamento do navegador."}
}
$("compose").onclick=buildPrompt;
$("save").onclick=saveProject;
$("generate").onclick=sendGeneration;
$("createCover").onclick=drawCover;
$("downloadCover").onclick=downloadCover;
$("connect").onclick=()=>{
 const endpoint=baseUrl();
 if(endpoint&&!/^https?:\/\//i.test(endpoint)){$("connectionStatus").textContent="Introduz um endereço HTTP ou HTTPS válido.";return}
 localStorage.setItem(ENDPOINT_KEY,endpoint);localStorage.setItem(TOKEN_KEY,$("apiToken").value.trim());testConnection();
};
$("endpoint").value=localStorage.getItem(ENDPOINT_KEY)||"";
$("apiToken").value=localStorage.getItem(TOKEN_KEY)||"";
renderLibrary();drawCover();
if("serviceWorker" in navigator)window.addEventListener("load",()=>navigator.serviceWorker.register("./sw.js").catch(()=>{}));
