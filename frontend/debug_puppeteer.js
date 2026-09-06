import puppeteer from 'puppeteer';
import { spawn } from 'child_process';
import fs from 'fs';

(async () => {
  const serverDir = 'C:\\Users\\urbra\\.gemini\\antigravity-cli\\brain\\ff104c7c-17d8-4b0d-bd9b-1bed7f58609b\\scratch\\server-room';
  const server = spawn('npm.cmd', ['run', 'start'], { cwd: serverDir, shell: true });
  
  server.stdout.on('data', data => console.log('SERVER: ' + data.toString()));
  server.stderr.on('data', data => console.log('SERVER ERR: ' + data.toString()));

  await new Promise(r => setTimeout(r, 3000));

  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  await page.goto('http://127.0.0.1:8080', { waitUntil: 'networkidle2' });
  const html = await page.evaluate(() => document.body.innerHTML);
  fs.writeFileSync('debug.html', html);
  
  await browser.close();
  server.kill();
})();
