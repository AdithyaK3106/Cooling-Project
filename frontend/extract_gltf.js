import puppeteer from 'puppeteer';
import { spawn, exec } from 'child_process';
import path from 'path';

(async () => {
  console.log('Starting HTTP server...');
  const serverDir = 'C:\\Users\\urbra\\.gemini\\antigravity-cli\\brain\\ff104c7c-17d8-4b0d-bd9b-1bed7f58609b\\scratch\\server-room';
  const server = spawn('npx', ['http-server', '-p', '8081', '-c-1', '.'], { cwd: serverDir, shell: true });
  
  server.stdout.on('data', d => console.log(d.toString()));
  server.stderr.on('data', d => console.error(d.toString()));

  await new Promise(r => setTimeout(r, 2000)); // wait for server to start

  console.log('Launching Puppeteer...');
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  const downloadPath = path.resolve('./public/models');
  const client = await page.target().createCDPSession();
  await client.send('Page.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: downloadPath,
  });

  console.log('Navigating to http://127.0.0.1:8081...');
  await page.goto('http://127.0.0.1:8081', { waitUntil: 'networkidle2', timeout: 60000 });
  
  console.log('Waiting for scene to build (10s)...');
  await new Promise(r => setTimeout(r, 10000));
  
  console.log('Extracting GLTF...');
  try {
    await page.evaluate(() => {
      // The plugin is instantiated as exportPugin inside the module,
      // but it exports a named room_server.gltf
      // Since we can't access exportPugin, we can click the button:
      const btn = document.getElementById('exportGltf');
      if (btn) btn.click();
    });
  } catch (e) {
    console.error('Extraction failed:', e);
  }
  
  console.log('Waiting for download to finish (10s)...');
  await new Promise(r => setTimeout(r, 10000));
  
  console.log('Done.');
  await browser.close();
  server.kill();
})();
