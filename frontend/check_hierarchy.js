import fs from 'fs';

const buffer = fs.readFileSync('public/models/room_server.glb');
const chunk0Length = buffer.readUInt32LE(12);
const jsonBuffer = buffer.subarray(20, 20 + chunk0Length);
const json = JSON.parse(jsonBuffer.toString('utf8'));

// Find "Rack 1" and "Rack body 1"
const rack1Idx = json.nodes.findIndex(n => n.name === 'Rack 1');
const rackBody1Idx = json.nodes.findIndex(n => n.name === 'Rack body 1');

console.log('Rack 1 Node:', json.nodes[rack1Idx]);
console.log('Rack body 1 Node:', json.nodes[rackBody1Idx]);

// Let's find who has rack body 1 as a child
const parentOfBody = json.nodes.find(n => n.children && n.children.includes(rackBody1Idx));
console.log('Parent of Rack body 1:', parentOfBody ? parentOfBody.name : 'NONE');

const parentOfRack1 = json.nodes.find(n => n.children && n.children.includes(rack1Idx));
console.log('Parent of Rack 1:', parentOfRack1 ? parentOfRack1.name : 'NONE');
