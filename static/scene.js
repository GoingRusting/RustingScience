(()=>{var re="attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}",ae=`precision highp float;
uniform vec2 uRes; uniform vec2 uHole; uniform float uScale; uniform float uTime; uniform vec2 uPar;

#define BH_SIZE 2.0
#define BH_SPIN 1.0
#define BH_BRIGHTNESS 1.0
#define END_NEBULA 1.0

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
float hash13(vec3 p3) {
  p3 = fract(p3 * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash13(i), hash13(i + vec3(1.0, 0.0, 0.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 0.0)), hash13(i + vec3(1.0, 1.0, 0.0)), f.x), f.y),
             mix(mix(hash13(i + vec3(0.0, 0.0, 1.0)), hash13(i + vec3(1.0, 0.0, 1.0)), f.x),
                 mix(hash13(i + vec3(0.0, 1.0, 1.0)), hash13(i + vec3(1.0, 1.0, 1.0)), f.x), f.y), f.z);
}
float fbm3(vec3 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += vnoise3(p) * a; p = p * 2.03 + vec3(1.7, 9.2, 3.1); a *= 0.5; }
  return s / 0.9375;
}
vec3 endHoleDir() { return normalize(vec3(0.45, 0.3, -0.84)); }

vec3 endSpace(vec3 d) {
  float t = uTime * 0.004;
  vec3 q = d * 2.0;
  vec3 w = vec3(fbm3(q + t), fbm3(q + 5.2 - t), fbm3(q + 9.7));
  float n    = fbm3(q * 1.4 + w * 2.2);
  float dust = fbm3(q * 3.1 + w * 3.0 + 20.0);
  vec3 neb = mix(vec3(0.22, 0.04, 0.42), vec3(0.03, 0.28, 0.5), smoothstep(0.35, 0.65, w.x));
  neb = mix(neb, vec3(0.85, 0.18, 0.5), smoothstep(0.55, 0.75, w.y));
  float dens = smoothstep(0.4, 0.85, n) * (1.0 - smoothstep(0.45, 0.7, dust) * 0.9);
  vec3 col = neb * dens * dens * 0.35 * END_NEBULA + vec3(0.002, 0.001, 0.004);
  for (int i = 0; i < 2; i++) {
    float scale = i == 0 ? 90.0 : 220.0;
    vec3 sp = d * scale;
    vec3 cell = floor(sp);
    float h = hash13(cell + float(i) * 13.0);
    const float th = 0.985;
    if (h > th) {
      vec3 c = vec3(hash13(cell + 1.3), hash13(cell + 7.1), hash13(cell + 3.7)) - 0.5;
      vec3 f = fract(sp) - 0.5 - c * 0.6;
      float s = exp(-dot(f, f) * 50.0) * (h - th) / (1.0 - th);
      vec3 tint = mix(vec3(1.0, 0.75, 0.55), vec3(0.65, 0.8, 1.0), hash13(cell + 5.0));
      col += tint * s * (i == 0 ? 6.0 : 2.0) * (1.0 - dust * 0.7);
    }
  }
  return col;
}

vec3 diskGlow(float r, float ang, float side) {
  if (r < 1.3 || r > 7.0) return vec3(0.0);
  float t = uTime * BH_SPIN;
  float a = ang + t * 1.2 / pow(r, 1.5) + log(r) * 2.5;
  float n = vnoise3(vec3(cos(a) * 3.0, sin(a) * 3.0, r * 2.5)) * 0.55
          + vnoise3(vec3(cos(a) * 9.0, sin(a) * 9.0, r * 9.0)) * 0.3
          + vnoise3(vec3(cos(a) * 20.0, sin(a) * 20.0, r * 30.0)) * 0.15;
  float lanes = 0.75 + 0.25 * sin(r * 18.0 + n * 9.0);
  float I = smoothstep(1.3, 1.7, r) * (1.0 - smoothstep(3.0, 7.0, r)) / (r * r) * 3.0;
  I *= (0.15 + 1.6 * n * n * n) * lanes;
  float x = smoothstep(1.5, 5.0, r);
  vec3 temp = mix(vec3(1.0, 0.92, 0.85), vec3(1.0, 0.42, 0.1), smoothstep(0.0, 0.5, x));
  temp = mix(temp, vec3(0.6, 0.08, 0.12), smoothstep(0.5, 1.0, x));
  float beam = pow(1.0 + 0.55 * side, 3.0);
  temp = mix(temp, vec3(0.75, 0.85, 1.0), max(side, 0.0) * 0.35 * (1.0 - x));
  return temp * I * beam * 8.0 * BH_BRIGHTNESS;
}

vec3 endSky(vec3 d) {
  vec3  bh   = endHoleDir();
  float R    = 0.08 * BH_SIZE;
  float cosA = dot(d, bh);
  float a    = acos(clamp(cosA, -1.0, 1.0));
  vec3  perp = d - bh * cosA;
  float pl   = length(perp);
  perp = pl > 1e-5 ? perp / pl : vec3(0.0, 1.0, 0.0);

  float rE = R * 1.7;
  float a2 = a - rE * rE / max(a, 1e-4);
  vec3 col = endSpace(bh * cos(a2) + perp * sin(a2));
  col *= smoothstep(R * 1.0, R * 1.15, a);

  float cosMax  = cos(min(R * 8.0, 3.14));
  float cosHalf = cos(min(R * 4.0, 3.14));
  if (cosA > cosMax) {
    vec3 u = normalize(cross(bh, vec3(0.15, 1.0, 0.05)));
    vec3 v = cross(u, bh);
    vec2 q = vec2(dot(perp, u), dot(perp, v)) * a / R;
    float tilt = 0.18;
    float rr = length(q);
    col += vec3(1.0, 0.8, 0.6) * exp(-pow((rr - 1.12) / 0.025, 2.0)) * 2.0 * BH_BRIGHTNESS;
    if (rr > 1.1) {
      float hr = 1.3 + (rr - 1.15) * 4.0;
      vec3 halo = diskGlow(hr, atan(q.y, q.x) * 1.0 + 1.3, q.x / rr * 0.6);
      col += halo * smoothstep(1.1, 1.25, rr) * (0.3 + 0.7 * abs(q.y) / rr) * 0.7;
    }
    vec2 dp = vec2(q.x, q.y / tilt);
    float r = length(dp);
    vec3 front = diskGlow(r, atan(dp.y, dp.x), dp.x / max(r, 1e-3));
    float behind = dp.y > 0.0 ? smoothstep(1.0, 1.15, rr) : 1.0;
    front *= behind;
    float cover = clamp(luminance(front) * 2.0, 0.0, 1.0);
    col = col * (1.0 - cover * 0.8) + front;
    col += vec3(0.5, 0.2, 0.35) * exp(-rr * 0.5) * 0.12 * BH_BRIGHTNESS * smoothstep(1.0, 1.15, rr) * smoothstep(cosMax, cosHalf, cosA);
  }
  return col;
}

vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }

void main() {
  // Camera looks straight at the hole (nudged by the pointer); the hole lands on uHole.
  vec3 f = endHoleDir();
  vec3 right = normalize(cross(f, vec3(0.0, 1.0, 0.0)));
  vec3 up = cross(right, f);
  f = normalize(f - right * uPar.x * 0.05 + up * uPar.y * 0.03);
  right = normalize(cross(f, vec3(0.0, 1.0, 0.0)));
  up = cross(right, f);
  vec2 px = (gl_FragCoord.xy - uHole) / uScale * 0.08 * BH_SIZE;
  vec3 d = normalize(f + right * px.x + up * px.y);

  vec3 color = endSky(d);
  // final.fsh: auto exposure in the End (eye brightness 0), ACES, gamma, grade.
  color = aces(color * 2.2);
  color = pow(color, vec3(1.0 / 2.2));
  color = mix(vec3(luminance(color)), color, 1.15);
  color = (color - 0.5) * 1.05 + 0.5;
  vec3 grade = mix(vec3(0.93, 1.0, 1.06), vec3(1.04, 1.0, 0.95), smoothstep(0.1, 0.75, luminance(color)));
  color *= grade;
  color += (ign(gl_FragCoord.xy) - 0.5) / 255.0;
  gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}`;function Q(r,s){let t=r.getContext("webgl",{antialias:!1,alpha:!1,powerPreference:"low-power"});if(!t)return null;let y=(h,E)=>{let B=t.createShader(h);if(t.shaderSource(B,E),t.compileShader(B),!t.getShaderParameter(B,t.COMPILE_STATUS))throw new Error(t.getShaderInfoLog(B)||"shader");return B},a=t.createProgram();t.attachShader(a,y(t.VERTEX_SHADER,re)),t.attachShader(a,y(t.FRAGMENT_SHADER,ae)),t.linkProgram(a),t.useProgram(a),t.bindBuffer(t.ARRAY_BUFFER,t.createBuffer()),t.bufferData(t.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),t.STATIC_DRAW),t.enableVertexAttribArray(0),t.vertexAttribPointer(0,2,t.FLOAT,!1,0,0);let e=h=>t.getUniformLocation(a,h),i=e("uRes"),f=e("uHole"),l=e("uScale"),g=e("uTime"),o=e("uPar"),n=1,p=1,c=0,u=!0,v=!0,q=!!s.still||matchMedia("(prefers-reduced-motion: reduce)").matches,I={x:0,y:0},_={x:0,y:0},L=performance.now()-(s.seed||0)*1e3,D=h=>{let E=q?4+(s.seed||0):(h-L)/1e3;I.x+=(_.x-I.x)*.04,I.y+=(_.y-I.y)*.04;let B=r.width/n,k=p>=300,d=k&&n<720,m=n*(d?.5:s.focusX||.68),b=k?d&&p>500?150:p*.42:p*.36,S=k?d?n*.05:Math.min(n*.034,p*.068):p*.1;t.viewport(0,0,r.width,r.height),t.uniform2f(i,r.width,r.height),t.uniform2f(f,m*B,(p-b)*B),t.uniform1f(l,S*B),t.uniform1f(g,E),t.uniform2f(o,I.x,I.y),t.drawArrays(t.TRIANGLES,0,3)},G=()=>{let h=r.getBoundingClientRect(),E=Math.min(devicePixelRatio||1,1);n=Math.max(1,h.width),p=Math.max(1,h.height),r.width=Math.round(n*E),r.height=Math.round(p*E),D(performance.now())},V=h=>{u&&(v&&!document.hidden&&D(h),c=requestAnimationFrame(V))},w=h=>{s.interactive!==!1&&(_.x=h.clientX/innerWidth-.5,_.y=h.clientY/innerHeight-.5)};G();let O=new ResizeObserver(G);O.observe(r);let A=new IntersectionObserver(h=>{v=h[0].isIntersecting});return A.observe(r),addEventListener("pointermove",w,{passive:!0}),q||(c=requestAnimationFrame(V)),()=>{u=!1,cancelAnimationFrame(c),O.disconnect(),A.disconnect(),removeEventListener("pointermove",w)}}function $(r,s){if(r=(r||"").trim(),!r)return s;if(r[0]==="#"){let y=r.slice(1);(y.length===3||y.length===4)&&(y=y.split("").map(e=>e+e).join(""));let a=parseInt(y.slice(0,6),16);return[a>>16&255,a>>8&255,a&255]}let t=r.match(/[\d.]+/g);return t&&t.length>=3?[+t[0],+t[1],+t[2]]:s}var C=(r,s)=>`rgba(${r[0]|0},${r[1]|0},${r[2]|0},${(s<0?0:s>1?1:s).toFixed(3)})`,Y=(r,s,t)=>[r[0]+(s[0]-r[0])*t,r[1]+(s[1]-r[1])*t,r[2]+(s[2]-r[2])*t];function j(r,s,t,y){let a=r[0],e=r[1],i=r[2],f,l,g;return y&&(f=Math.cos(y),l=Math.sin(y),g=a*f-e*l,e=a*l+e*f,a=g),f=Math.cos(s),l=Math.sin(s),g=a*f+i*l,i=-a*l+i*f,a=g,f=Math.cos(t),l=Math.sin(t),g=e*f-i*l,i=e*l+i*f,e=g,[a,e,i]}function K(r){let s=r>>>0||1;return()=>(s^=s<<13,s^=s>>>17,s^=s<<5,(s>>>0)%1e5/1e5)}function se(r,s){let t=r.length,y=[],a=[],e=(i,f)=>{let l=f[0]-i[0],g=f[1]-i[1],o=Math.hypot(l,g);return[-g/o,l/o]};for(let i=0;i<t;i++)if(i===0||i===t-1){let f=i===0?e(r[0],r[1]):e(r[t-2],r[t-1]);y.push([r[i][0]+f[0]*s,r[i][1]+f[1]*s]),a.push([r[i][0]-f[0]*s,r[i][1]-f[1]*s])}else{let f=e(r[i-1],r[i]),l=e(r[i],r[i+1]),g=[f[0]+l[0],f[1]+l[1]],o=Math.hypot(g[0],g[1]);g=[g[0]/o,g[1]/o];let n=s/(g[0]*f[0]+g[1]*f[1]);y.push([r[i][0]+g[0]*n,r[i][1]+g[1]*n]),a.push([r[i][0]-g[0]*n,r[i][1]-g[1]*n])}return y.concat(a.reverse())}function x(r,s,t){let y=r.map(i=>[i[0]-128,128-i[1],s]),a=r.map(i=>[i[0]-128,128-i[1],t]),e=[];for(let i=0;i<r.length;i++){let f=(i+1)%r.length;e.push([y[f],y[i],a[i],a[f]])}return{front:y,back:a.slice().reverse(),sides:e}}function ee(){let r=x(se([[235,32],[92,32],[36,88],[36,168],[92,224],[220,224],[220,128],[137,128]],15),-18,18),s=x([[186,17],[220,17],[220,47],[186,47]],-18.6,18),t=K(7),y=[];for(let a=0;a<70;a++)y.push({a:t()*6.283,rad:150+t()*120,y:(t()-.5)*220,sp:.08+t()*.22,s:.6+t()*1.6,tw:t()*6});return(a,e,i,f,l,g,o)=>{let n=i<760,p=n?i*.5:i*(o.focusX||.7),c=f*(n?.3:o.focusY||.46),u=Math.min(f*(o.fill||.0026),i*(n?.0026:.0019)),v=900,q=900,I=Math.sin(e*.35)*.5+g.x*.5-.18,_=Math.sin(e*.27)*.1+g.y*.25-.08,L=Math.sin(e*.21)*.03,D=Math.sin(e*.8)*6,G=R=>{let M=j(R,I,_,L),P=M[2]*u+v;return[p+M[0]*u*q/P,c-(M[1]+D)*u*q/P,P,M]};a.lineWidth=1;let V=-190;for(let R=-8;R<=8;R++){let M=G([R*50,V,-300]),P=G([R*50,V,500]),z=a.createLinearGradient(M[0],M[1],P[0],P[1]);z.addColorStop(0,C(l.accent,0)),z.addColorStop(.35,C(l.accent,.16)),z.addColorStop(1,C(l.accent,0)),a.strokeStyle=z,a.beginPath(),a.moveTo(M[0],M[1]),a.lineTo(P[0],P[1]),a.stroke()}for(let R=-300;R<=500;R+=50){let M=G([-400,V,R]),P=G([400,V,R]);a.strokeStyle=C(l.accent,.13*(1-(R+300)/800)),a.beginPath(),a.moveTo(M[0],M[1]),a.lineTo(P[0],P[1]),a.stroke()}let w=G([0,0,0]),O=a.createRadialGradient(w[0],w[1],0,w[0],w[1],260*u*1.4);O.addColorStop(0,C(l.accent,.22)),O.addColorStop(1,C(l.accent,0)),a.fillStyle=O,a.fillRect(0,0,i,f);let A=R=>{a.globalCompositeOperation="lighter";for(let M=0;M<y.length;M++){let P=y[M],z=P.a+e*P.sp,T=G([Math.cos(z)*P.rad,P.y+Math.sin(e*.5+P.tw)*10,Math.sin(z)*P.rad]);if(T[2]>v!==R)continue;let F=.35+.35*Math.sin(e*2+P.tw);a.fillStyle=C(M%5===0?l.hot:l.accent,F),a.beginPath(),a.arc(T[0],T[1],P.s*(R?.8:1.2),0,6.283),a.fill()}a.globalCompositeOperation="source-over"};A(!0);let h=[-.45,.6,-.65],E=(R,M,P,z)=>{let T=R.map(G),F=0;for(let N=0;N<T.length;N++){let X=(N+1)%T.length;F+=T[N][0]*T[X][1]-T[X][0]*T[N][1]}return{pts:T,area:F,z:T.reduce((N,X)=>N+X[2],0)/T.length,base:M,edge:P,edgeA:z}},B=E(r.front,l.accent,l.hot,.25),k=B.area>0?1:-1,d=[];[r,s].forEach((R,M)=>{R.sides.forEach(P=>{let z=E(P,M?l.hot:l.accent,l.accent,.2);if(z.area*k<=0)return;let T=z.pts[0][3],F=z.pts[1][3],N=z.pts[2][3],X=[F[0]-T[0],F[1]-T[1],F[2]-T[2]],U=[N[0]-T[0],N[1]-T[1],N[2]-T[2]],Z=X[1]*U[2]-X[2]*U[1],J=X[2]*U[0]-X[0]*U[2],W=X[0]*U[1]-X[1]*U[0],oe=Math.hypot(Z,J,W)||1,ne=Math.abs((Z*h[0]+J*h[1]+W*h[2])/oe);z.base=Y(M?Y(l.hot,l.deep,.35):l.deep,M?l.hot:l.accent,.15+ne*.6),d.push(z)})}),d.sort((R,M)=>M.z-R.z);let m=R=>{a.beginPath(),R.pts.forEach((M,P)=>{P?a.lineTo(M[0],M[1]):a.moveTo(M[0],M[1])}),a.closePath()},b=(R,M)=>{m(R),a.fillStyle=C(R.base,M),a.fill(),a.strokeStyle=C(R.edge,R.edgeA),a.lineWidth=1,a.stroke()};d.forEach(R=>b(R,1)),a.save(),m(B),a.fillStyle=C(l.accent,1),a.fill(),a.clip();let S=w[0]+Math.sin(e*.5)*220*u,H=a.createLinearGradient(S-120*u,0,S+120*u,f);H.addColorStop(0,"rgba(255,255,255,0)"),H.addColorStop(.5,"rgba(255,255,255,0.14)"),H.addColorStop(1,"rgba(255,255,255,0)"),a.fillStyle=H,a.fillRect(0,0,i,f),a.restore(),b(E(s.front,l.hot,l.hot,0),1),A(!1)}}function ie(r){let s=(1+Math.sqrt(5))/2,t=[];[[0,1,s],[1,s,0],[s,0,1]].forEach(n=>{[1,-1].forEach(p=>{[1,-1].forEach(c=>{let u=n.slice(),v=[];u.forEach((q,I)=>{q&&v.push(I)}),u[v[0]]*=p,u[v[1]]*=c,t.push(u)})})});let y=[];for(let n=0;n<t.length;n++)for(let p=n+1;p<t.length;p++){let c=Math.hypot(t[n][0]-t[p][0],t[n][1]-t[p][1],t[n][2]-t[p][2]);Math.abs(c-2)<.01&&y.push([n,p])}let a=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],e=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],i=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],f=[[0,2],[0,3],[0,4],[0,5],[1,2],[1,3],[1,4],[1,5],[2,4],[2,5],[3,4],[3,5]],l=K(11),g=Math.round((r.density||1)*130),o=[];for(let n=0;n<g;n++)o.push({x:(l()-.5)*1e3,y:l()*700,z:-150+l()*650,s:5+l()*9,v:30+l()*60,ax:l()*6,ay:l()*6,w:(l()-.5)*2});return(n,p,c,u,v,q,I)=>{let _=c<760,L=c*(_?.5:I.focusX||.68),D=u*.48,G=Math.min(c,u)/700,V=1e3,w=900,O=p*.12+q.x*.4,A=-.32+q.y*.2,h=m=>{let b=j(m,O,A),S=b[2]+V;return[L+b[0]*w/S*G,D-b[1]*w/S*G,S]};n.lineWidth=1;for(let m=-6;m<=6;m++){let b=h([m*70,-260,-420]),S=h([m*70,-260,420]);n.strokeStyle=C(v.accent,.08),n.beginPath(),n.moveTo(b[0],b[1]),n.lineTo(S[0],S[1]),n.stroke(),b=h([-420,-260,m*70]),S=h([420,-260,m*70]),n.beginPath(),n.moveTo(b[0],b[1]),n.lineTo(S[0],S[1]),n.stroke()}let E=[];for(let m=0;m<o.length;m++){let b=o[m],S=350-(b.y+p*b.v)%700,H=S<-250;H&&(S=-250+b.s);let R=b.ay+p*b.w*(H?0:1),M=b.ax+p*b.w*(H?0:.7),P=a.map(z=>{let T=j([z[0]*b.s,z[1]*b.s,z[2]*b.s],R,M);return h([T[0]+b.x,T[1]+S,T[2]+b.z])});E.push({pts:P,z:P[0][2],hot:m%9===0})}E.sort((m,b)=>b.z-m.z),E.forEach(m=>{let b=Math.max(0,Math.min(1,(1700-m.z)/1100));n.strokeStyle=C(m.hot?v.hot:v.accent,.12+b*.5),n.beginPath(),e.forEach(S=>{n.moveTo(m.pts[S[0]][0],m.pts[S[0]][1]),n.lineTo(m.pts[S[1]][0],m.pts[S[1]][1])}),n.stroke()});let B=(m,b,S,H,R,M,P,z)=>{let T=m.map(F=>h(j([F[0]*S,F[1]*S,F[2]*S],H,R)));return b.forEach(F=>{let N=T[F[0]],X=T[F[1]],U=Math.max(0,Math.min(1,(V+S-(N[2]+X[2])/2)/(2*S)));n.strokeStyle=C(M,P*(.25+U*.75)),n.lineWidth=z,n.beginPath(),n.moveTo(N[0],N[1]),n.lineTo(X[0],X[1]),n.stroke()}),T},k=n.createRadialGradient(L,D,0,L,D,300*G);k.addColorStop(0,C(v.accent,.18)),k.addColorStop(1,C(v.accent,0)),n.fillStyle=k,n.fillRect(0,0,c,u),n.globalCompositeOperation="lighter";let d=B(t,y,125,p*.25,p*.15,v.accent,.9,1.4);B(i,f,120,-p*.4,.6,v.hot,.55,1),B(a,e,52,p*.7,p*.5,v.hot,.9,1.6),d.forEach(m=>{n.fillStyle=C(v.hot,.9),n.fillRect(m[0]-2,m[1]-2,4,4)}),n.globalCompositeOperation="source-over"}}function le(r){let s=K(23),t=[],y=[],a=[],e=6,i=Math.round(18*(r.density||1));for(let o=0;o<e;o++)for(let n=0;n<i;n++){let p=s()*6.283,c=Math.sqrt(s())*(200-Math.abs(o-2.5)*26),u=(o-(e-1)/2)*85;t.push({p:[u+(s()-.5)*34-c*c*.0012,Math.sin(p)*c*.8,Math.cos(p)*c],l:o,act:0})}t.forEach(()=>a.push([])),t.forEach((o,n)=>{let p=[];t.forEach((c,u)=>{if(c.l===o.l+1||c.l===o.l&&u>n){let v=Math.hypot(o.p[0]-c.p[0],o.p[1]-c.p[1],o.p[2]-c.p[2]);p.push([v+(c.l===o.l?60:0),u])}}),p.sort((c,u)=>c[0]-u[0]),p.slice(0,o.l===e-1?1:3).forEach(c=>{let u=y.length;y.push([n,c[1]]),t[c[1]].l>o.l&&a[n].push(u)})});let f=[],l=()=>{let o=Math.floor(s()*i),n=a[o][Math.floor(s()*a[o].length)];n!==void 0&&f.push({e:n,u:0,v:.5+s()*.7})};for(let o=0;o<22;o++)l(),f[f.length-1]&&(f[f.length-1].u=s());let g=0;return(o,n,p,c,u,v,q)=>{let I=Math.min(.05,Math.max(0,n-g));g=n;let _=p<760,L=p*(_?.5:q.focusX||.68),D=c*.5,G=Math.min(p*(_?1.3:.95),c*1.45)/560,V=900,w=800,O=n*.1+.5+v.x*.5,A=.18+v.y*.25+Math.sin(n*.2)*.06,h=t.map(k=>{let d=j(k.p,O,A),m=d[2]+V;return[L+d[0]*w/m*G,D-d[1]*w/m*G,m]}),E=o.createRadialGradient(L,D,0,L,D,340*G);E.addColorStop(0,C(u.accent,.14)),E.addColorStop(1,C(u.accent,0)),o.fillStyle=E,o.fillRect(0,0,p,c),o.lineWidth=1,y.forEach(k=>{let d=h[k[0]],m=h[k[1]],b=Math.max(0,Math.min(1,(V+300-(d[2]+m[2])/2)/600));o.strokeStyle=C(u.accent,.05+b*.2),o.beginPath(),o.moveTo(d[0],d[1]),o.lineTo(m[0],m[1]),o.stroke()}),o.globalCompositeOperation="lighter";for(let k=f.length-1;k>=0;k--){let d=f[k];d.u+=I*d.v;let m=y[d.e],b=h[m[0]],S=h[m[1]];if(d.u>=1){t[m[1]].act=1;let F=a[m[1]];F.length&&s()<.92?(d.e=F[Math.floor(s()*F.length)],d.u=0):(f.splice(k,1),l());continue}let H=b[0]+(S[0]-b[0])*d.u,R=b[1]+(S[1]-b[1])*d.u,M=Math.max(0,d.u-.25),P=b[0]+(S[0]-b[0])*M,z=b[1]+(S[1]-b[1])*M,T=o.createLinearGradient(P,z,H,R);T.addColorStop(0,C(u.hot,0)),T.addColorStop(1,C(u.hot,.9)),o.strokeStyle=T,o.lineWidth=2,o.beginPath(),o.moveTo(P,z),o.lineTo(H,R),o.stroke(),o.fillStyle=C(u.hot,1),o.beginPath(),o.arc(H,R,2.2,0,6.283),o.fill()}o.globalCompositeOperation="source-over",h.map((k,d)=>d).sort((k,d)=>h[d][2]-h[k][2]).forEach(k=>{let d=h[k],m=t[k],b=Math.max(0,Math.min(1,(V+300-d[2])/600));m.act*=Math.pow(.12,I);let S=(1.8+b*2.6)*(w/d[2])*Math.max(.8,G);if(m.act>.05){let H=o.createRadialGradient(d[0],d[1],0,d[0],d[1],S*6);H.addColorStop(0,C(u.hot,.5*m.act)),H.addColorStop(1,C(u.hot,0)),o.fillStyle=H,o.beginPath(),o.arc(d[0],d[1],S*6,0,6.283),o.fill()}o.fillStyle=C(Y(u.bg,u.accent,.35+b*.65),1),o.beginPath(),o.arc(d[0],d[1],S,0,6.283),o.fill(),o.fillStyle=C(Y(u.accent,u.hot,m.act),.5+b*.5),o.beginPath(),o.arc(d[0],d[1],S*.45,0,6.283),o.fill()})}}function ce(r){let s=K(5),t=[],y=[],a=Math.round(900*(r.density||1));for(let e=0;e<260;e++)t.push({x:s(),y:s(),s:s()<.9?.8:1.5,tw:s()*6.283,b:.25+s()*.6});for(let e=0;e<a;e++){let i=s();y.push({r:1.5+Math.pow(i,2.2)*3.4,a:s()*6.283,w:(s()-.5)*.05,s:.7+s()*1.4})}return(e,i,f,l,g,o,n)=>{let p=f<760,c=f*(p?.5:n.focusX||.68)+o.x*14,u=l*.47+o.y*10,v=Math.min(f*(p?.13:.085),l*.15),q=.16+o.y*.05+Math.sin(i*.15)*.015,I=-.14+o.x*.06,_=Math.cos(I),L=Math.sin(I),D=(A,h)=>[c+A*_-h*L,u+A*L+h*_];for(let A=0;A<t.length;A++){let h=t[A],E=h.x*f-c,B=h.y*l-u,k=Math.hypot(E,B)||1,d=(k+v*v*1.8/k)/k;E*=d,B*=d,!(Math.hypot(E,B)<v*1.2)&&(e.fillStyle=C(g.ink,h.b*(.6+.4*Math.sin(i*1.3+h.tw))),e.fillRect(c+E,u+B,h.s,h.s))}let G=e.createRadialGradient(c,u,v,c,u,v*6);G.addColorStop(0,C(g.accent,.3)),G.addColorStop(.35,C(g.accent,.09)),G.addColorStop(1,C(g.accent,0)),e.fillStyle=G,e.fillRect(0,0,f,l);let V=()=>{let A=e.createLinearGradient(c-v*5,0,c+v*5,0);return A.addColorStop(0,C(g.hot,1)),A.addColorStop(.45,C(Y(g.hot,g.accent,.4),.9)),A.addColorStop(1,C(g.accent,.45)),A},w=(A,h)=>{e.save(),e.translate(c,u),e.rotate(I);for(let E=0;E<30;E++){let B=v*(1.5+E*.11);e.globalAlpha=h*Math.pow(1-E/30,1.8)*.32,e.strokeStyle=V(),e.lineWidth=Math.max(1,v*.06),e.beginPath(),e.ellipse(0,0,B,B*q,0,A>0?0:Math.PI,A>0?Math.PI:2*Math.PI),e.stroke()}e.restore(),e.globalAlpha=1},O=A=>{for(let h=0;h<y.length;h++){let E=y[h],B=E.a+i*.9*Math.pow(E.r,-1.5);if(Math.sin(B)<0!==A)continue;let k=E.r*v,d=D(Math.cos(B)*k,Math.sin(B)*k*q+E.w*v),m=Math.min(1,(E.r-1.5)/2.4),b=1+.6*Math.cos(B+.3)*-1;e.fillStyle=C(Y(g.hot,g.accent,m),Math.min(1,(.9-m*.6)*b*.6)),e.fillRect(d[0],d[1],E.s,E.s)}};e.globalCompositeOperation="lighter",w(-1,.8),O(!0),e.save(),e.translate(c,u),e.rotate(I);for(let A=0;A<22;A++){let h=v*(1.1+A*.026);e.globalAlpha=Math.pow(1-A/22,1.6)*.42,e.strokeStyle=V(),e.lineWidth=Math.max(1,v*.03),e.beginPath(),e.ellipse(0,0,h,h*.96,0,Math.PI*1.02,Math.PI*1.98),e.stroke(),e.globalAlpha*=.45,e.beginPath(),e.ellipse(0,0,h*.98,h*.9,0,Math.PI*.12,Math.PI*.88),e.stroke()}e.restore(),e.globalAlpha=1,e.globalCompositeOperation="source-over",e.fillStyle="rgb(4,6,9)",e.beginPath(),e.arc(c,u,v,0,6.283),e.fill(),e.globalCompositeOperation="lighter",e.lineWidth=Math.max(1.2,v*.03),e.strokeStyle=C(g.hot,.9),e.beginPath(),e.arc(c,u,v*1.03,0,6.283),e.stroke(),e.lineWidth=v*.1,e.strokeStyle=C(g.accent,.16),e.beginPath(),e.arc(c,u,v*1.09,0,6.283),e.stroke(),w(1,1),O(!1),e.globalCompositeOperation="source-over"}}var he={forge:ee,engine:ie,brain:le,shader:ce};function te(r,s,t){if(s==="shader"){let w=Q(r,t);if(w)return w}let y=r.getContext("2d");if(!y)return()=>{};let a=(he[s]||ee)(t),e=1,i=1,f=0,l=!0,g=!0,o=!!t.still||matchMedia("(prefers-reduced-motion: reduce)").matches,n={x:0,y:0},p={x:0,y:0},c={},u=performance.now()-(t.seed||0)*1e3,v=()=>{let w=getComputedStyle(r);c.accent=$(w.getPropertyValue("--accent"),[239,146,94]),c.hot=$(w.getPropertyValue("--scene-hot"),[245,238,227]),c.ink=$(w.getPropertyValue("--ink"),[245,238,227]),c.bg=$(w.getPropertyValue("--surface-000"),[15,21,27]),c.deep=Y(c.bg,c.accent,.42)},q=w=>{let O=o?4+(t.seed||0):(w-u)/1e3;n.x+=(p.x-n.x)*.04,n.y+=(p.y-n.y)*.04,y.clearRect(0,0,e,i),a(y,O,e,i,c,n,t)},I=()=>{let w=r.getBoundingClientRect(),O=Math.min(devicePixelRatio||1,2);e=Math.max(1,w.width),i=Math.max(1,w.height),r.width=Math.round(e*O),r.height=Math.round(i*O),y.setTransform(O,0,0,O,0,0),o&&q(performance.now())},_=w=>{l&&(g&&!document.hidden&&q(w),f=requestAnimationFrame(_))},L=w=>{t.interactive!==!1&&(p.x=w.clientX/innerWidth-.5,p.y=w.clientY/innerHeight-.5)};v(),I();let D=new ResizeObserver(I);D.observe(r);let G=new IntersectionObserver(w=>{g=w[0].isIntersecting});G.observe(r);let V=setInterval(()=>{v(),o&&q(performance.now())},600);return addEventListener("pointermove",L,{passive:!0}),o||(f=requestAnimationFrame(_)),()=>{l=!1,cancelAnimationFrame(f),clearInterval(V),D.disconnect(),G.disconnect(),removeEventListener("pointermove",L)}}for(let r of document.querySelectorAll(".gr-scene"))te(r.querySelector("canvas"),r.dataset.kind,JSON.parse(r.dataset.opt||"{}"));})();
