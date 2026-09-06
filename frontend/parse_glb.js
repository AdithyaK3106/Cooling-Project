import fs from 'fs';

const buffer = fs.readFileSync('public/models/room_server.glb');
// The first 12 bytes are the GLB header: magic (4), version (4), length (4)
const magic = buffer.toString('utf8', 0, 4);
if (magic !== 'glTF') {
  console.error('Not a valid GLB');
  process.exit(1);
}

// Chunk 0 is the JSON chunk
const chunk0Length = buffer.readUInt32LE(12);
const chunk0Type = buffer.toString('utf8', 16, 20);
if (chunk0Type !== 'JSON') {
  console.error('First chunk is not JSON');
  process.exit(1);
}

const jsonBuffer = buffer.subarray(20, 20 + chunk0Length);
const json = JSON.parse(jsonBuffer.toString('utf8'));

const nodeNames = json.nodes.map(n => n.name).filter(n => n);
console.log('Unique node names:', [...new Set(nodeNames)]);
