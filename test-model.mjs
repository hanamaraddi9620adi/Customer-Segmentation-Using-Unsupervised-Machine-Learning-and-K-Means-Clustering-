import fs from 'node:fs';import assert from 'node:assert/strict';import {predict} from './public/model.js';
const d=JSON.parse(fs.readFileSync(new URL('./public/data.json',import.meta.url),'utf8'));
for(const c of d.customers)assert.equal(predict(d.model.features.map(f=>c[f]),d.model).cluster,c.cluster);
for(const bad of [[-1,1,1,1,1,1],[NaN,1,1,1,1,1],[Infinity,1,1,1,1,1],[]])assert.throws(()=>predict(bad,d.model));
console.log('PASS: browser inference matches Python for all 440 customers; invalid inputs rejected.');
