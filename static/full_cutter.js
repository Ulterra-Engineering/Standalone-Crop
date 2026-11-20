(() => {
  const $ = id => document.getElementById(id);
  const image = $("fc-image"), canvas = $("fc-canvas"), ctx = canvas.getContext("2d");
  const state = { images: [], index: 0, view: "", boxes: [], selected: -1, dirty: false, pointer: null };

  const current = () => state.images[state.index];
  const setStatus = message => { $("fc-status").textContent = message; };
  const setDirty = () => { state.dirty = true; setStatus("Unsaved changes."); };
  function boxAt(x, y) {
    for (let i = state.boxes.length - 1; i >= 0; i--) {
      const b = state.boxes[i];
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) return i;
    }
    return -1;
  }
  function draw() {
    if (!image.naturalWidth) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.boxes.forEach((b, i) => {
      const x=b[0]*canvas.width, y=b[1]*canvas.height, w=(b[2]-b[0])*canvas.width, h=(b[3]-b[1])*canvas.height;
      const color=i===state.selected ? "#ffe14a" : "#36e0ff";
      ctx.fillStyle=i===state.selected ? "rgba(255,225,74,.18)" : "rgba(54,224,255,.13)";
      ctx.fillRect(x,y,w,h);
      ctx.lineWidth=Math.max(5,canvas.width/350); ctx.strokeStyle="rgba(0,0,0,.95)"; ctx.strokeRect(x,y,w,h);
      ctx.lineWidth=Math.max(2.5,canvas.width/650); ctx.strokeStyle=color; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle=color; ctx.font=`bold ${Math.max(14,canvas.width/55)}px system-ui`;
      ctx.fillText(String(i+1),x+5,y+20);
    });
    $("fc-box-count").textContent=state.boxes.length;
    $("fc-delete").disabled=state.selected<0;
  }
  function pointerPosition(event) {
    const rect=image.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  }
  function updateProgress() {
    const reviewed=state.images.filter(x=>x.reviewed).length;
    const annotated=state.images.filter(x=>x.status==="annotated").length;
    const skipped=state.images.filter(x=>x.status==="skip").length;
    $("fc-position").textContent=`${state.index+1} / ${state.images.length} · ${reviewed} reviewed`;
    $("fc-reviewed").textContent=`${annotated} annotated · ${skipped} skipped · ${state.images.length-reviewed} pending`;
  }
  function renderCurrent() {
    const item=current(); if(!item)return;
    state.view=item.view||""; state.boxes=(item.boxes||[]).map(b=>b.map(Number));
    state.selected=-1; state.dirty=false;
    $("fc-frame").hidden=true; $("fc-empty").hidden=false; $("fc-empty").textContent="Loading image…";
    image.src=`/api/full-cutter/image?key=${encodeURIComponent(item.key)}`;
    $("fc-name").textContent=`${item.name} · ${item.width} × ${item.height}`;
    $("fc-view-status").textContent=state.view ? `Assigned: ${state.view.toUpperCase()}` : "View not assigned";
    $("fc-view-top").classList.toggle("fc-active",state.view==="top");
    $("fc-view-isometric").classList.toggle("fc-active",state.view==="isometric");
    setStatus(item.status==="annotated" ? "Saved annotation loaded." : item.status==="skip" ? "Previously marked Skip." : "Not yet reviewed.");
    updateProgress(); draw();
  }
  image.addEventListener("load",()=>{
    canvas.width=image.naturalWidth; canvas.height=image.naturalHeight;
    $("fc-frame").hidden=false; $("fc-empty").hidden=true; draw();
  });
  image.addEventListener("error",()=>{
    $("fc-frame").hidden=true; $("fc-empty").hidden=false; $("fc-empty").textContent="Could not load this image.";
  });
  canvas.addEventListener("pointerdown",event=>{
    if(!image.naturalWidth)return;
    const [x,y]=pointerPosition(event),hit=boxAt(x,y); state.selected=hit;
    state.pointer={startX:x,startY:y,original:hit>=0?state.boxes[hit].slice():null,moved:false};
    canvas.setPointerCapture(event.pointerId); draw();
  });
  canvas.addEventListener("pointermove",event=>{
    if(!state.pointer)return;
    const [x,y]=pointerPosition(event),p=state.pointer;
    if(p.original){
      const dx=x-p.startX,dy=y-p.startY,b=p.original,w=b[2]-b[0],h=b[3]-b[1];
      if(Math.abs(dx)+Math.abs(dy)>.001)p.moved=true;
      const x1=Math.max(0,Math.min(1-w,b[0]+dx)),y1=Math.max(0,Math.min(1-h,b[1]+dy));
      state.boxes[state.selected]=[x1,y1,x1+w,y1+h];
    } else { p.endX=x;p.endY=y;p.moved=true; }
    draw();
    if(!p.original&&p.endX!==undefined){
      const r=[Math.min(p.startX,p.endX),Math.min(p.startY,p.endY),Math.max(p.startX,p.endX),Math.max(p.startY,p.endY)];
      ctx.strokeStyle="#36d7ff";ctx.lineWidth=Math.max(2,canvas.width/700);ctx.setLineDash([8,6]);
      ctx.strokeRect(r[0]*canvas.width,r[1]*canvas.height,(r[2]-r[0])*canvas.width,(r[3]-r[1])*canvas.height);ctx.setLineDash([]);
    }
  });
  function finishPointer(){
    if(!state.pointer)return;const p=state.pointer;
    if(!p.original&&p.endX!==undefined){
      const b=[Math.min(p.startX,p.endX),Math.min(p.startY,p.endY),Math.max(p.startX,p.endX),Math.max(p.startY,p.endY)];
      if(b[2]-b[0]>=.003&&b[3]-b[1]>=.003){state.boxes.push(b);state.selected=state.boxes.length-1;setDirty();}
    }else if(p.original&&p.moved)setDirty();
    state.pointer=null;draw();
  }
  canvas.addEventListener("pointerup",finishPointer);canvas.addEventListener("pointercancel",finishPointer);

  async function saveCurrent(status="annotated"){
    const item=current();if(!item)return true;
    if(!state.view){setStatus("Choose Top or Isometric before saving or skipping.");return false;}
    if(status==="annotated"&&!state.boxes.length){setStatus("Draw at least one full-cutter box, or choose Skip image.");return false;}
    const response=await fetch("/api/full-cutter/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:item.key,view:state.view,boxes:state.boxes,status})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||"Save failed.");
    item.view=state.view;item.boxes=state.boxes.map(b=>b.slice());item.status=status;
    item.annotated=status==="annotated";item.reviewed=true;state.dirty=false;
    updateProgress();setStatus(status==="skip"?"Image marked Skip and saved.":"Annotation saved to the separate full-cutter manifest.");
    return true;
  }
  async function move(delta){
    try{if(state.dirty&&!(await saveCurrent()))return;
      const next=state.index+delta;if(next<0||next>=state.images.length)return;
      state.index=next;renderCurrent();
    }catch(error){setStatus(error.message);}
  }
  $("fc-previous").addEventListener("click",()=>move(-1));$("fc-next").addEventListener("click",()=>move(1));
  $("fc-view-top").addEventListener("click",()=>{state.view="top";$("fc-view-status").textContent="Assigned: TOP";$("fc-view-top").classList.add("fc-active");$("fc-view-isometric").classList.remove("fc-active");setDirty();});
  $("fc-view-isometric").addEventListener("click",()=>{state.view="isometric";$("fc-view-status").textContent="Assigned: ISOMETRIC";$("fc-view-isometric").classList.add("fc-active");$("fc-view-top").classList.remove("fc-active");setDirty();});
  $("fc-delete").addEventListener("click",()=>{if(state.selected<0)return;state.boxes.splice(state.selected,1);state.selected=-1;setDirty();draw();});
  $("fc-clear").addEventListener("click",()=>{if(!state.boxes.length)return;state.boxes=[];state.selected=-1;setDirty();draw();});
  $("fc-save").addEventListener("click",async()=>{try{await saveCurrent("annotated");}catch(error){setStatus(error.message);}});
  $("fc-skip").addEventListener("click",async()=>{try{if(await saveCurrent("skip"))move(1);}catch(error){setStatus(error.message);}});
  document.addEventListener("keydown",event=>{
    if(event.key==="Delete"&&state.selected>=0)$("fc-delete").click();
    if(event.key==="ArrowLeft")move(-1);
    if(event.key==="ArrowRight")move(1);
  });
  fetch("/api/full-cutter/images").then(r=>r.json()).then(data=>{
    state.images=data.images||[];
    if(!state.images.length){$("fc-empty").hidden=false;$("fc-empty").textContent=data.error||"No images found in full_cutter/.";$("fc-position").textContent="0 images";return;}
    renderCurrent();
  }).catch(error=>{$("fc-empty").hidden=false;$("fc-empty").textContent=`Could not load image list: ${error.message}`;});
})();
