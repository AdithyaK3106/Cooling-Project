import fs from 'fs';

// Since we can't easily parse GLTF without three.js in node, let's write a small script that parses the scene hierarchy from `frontend/check_hierarchy.js`.
// Wait, I can just use Three.js and GLTFLoader in node if I have the right polyfills, or I can just use my frontend to print it.
