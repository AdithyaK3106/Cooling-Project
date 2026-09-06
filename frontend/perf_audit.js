import puppeteer from 'puppeteer';

(async () => {
  console.log("Starting Chrome...");
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  
  console.log("Navigating to dashboard...");
  const startTime = Date.now();
  await page.goto('http://localhost:5173/');
  
  // Wait for Canvas to render
  await page.waitForSelector('canvas', { timeout: 60000 });
  console.log(`Canvas loaded in ${Date.now() - startTime}ms`);
  
  // Wait a few seconds for the model to load
  await new Promise(r => setTimeout(r, 10000));
  
  const metrics = await page.evaluate(() => {
    if (!window.__gl) return { error: "No __gl exposed" };
    
    return {
      memory: window.__gl.info.memory,
      render: window.__gl.info.render,
      programs: window.__gl.info.programs ? window.__gl.info.programs.length : null,
    };
  });
  
  console.log("Metrics:", metrics);
  await browser.close();
})();
