import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import path from 'path';

(async () => {
  console.log('Starting HTTP server...');
  const serverDir = 'C:\\Users\\urbra\\.gemini\\antigravity-cli\\brain\\ff104c7c-17d8-4b0d-bd9b-1bed7f58609b\\scratch\\server-room';
  const server = spawn('npx', ['http-server', '-p', '8080', '-c-1', '.'], { cwd: serverDir, shell: true });

  await new Promise(r => setTimeout(r, 2000)); // wait for server to start

  console.log('Launching Puppeteer...');
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  const downloadPath = path.resolve('./public');
  const client = await page.target().createCDPSession();
  await client.send('Page.setDownloadBehavior', {
    behavior: 'allow',
    downloadPath: downloadPath,
  });

  console.log('Navigating to http://127.0.0.1:8080...');
  await page.goto('http://127.0.0.1:8080', { waitUntil: 'networkidle2', timeout: 60000 });
  
  console.log('Waiting for scene to build (10s)...');
  await new Promise(r => setTimeout(r, 10000));
  
  console.log('Clicking export button...');
  await page.evaluate(() => {
    document.getElementById('exportGltf').click();
  });
  
  console.log('Waiting for download to finish (10s)...');
  await new Promise(r => setTimeout(r, 10000));
  
  console.log('Done.');
  await browser.close();
  server.kill();
})();
