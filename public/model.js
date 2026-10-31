export function predict(values, model) {
 if(values.length!==model.features.length || values.some(v=>!Number.isFinite(v)||v<0)) throw new Error('Enter a finite, nonnegative amount in every category.');
 const z=values.map((v,i)=>(Math.log1p(v)-model.mean[i])/model.scale[i]);
 const distances=model.centers.map(c=>Math.sqrt(c.reduce((s,v,i)=>s+(v-z[i])**2,0)));
 return {cluster:distances.indexOf(Math.min(...distances)),distances};
}
