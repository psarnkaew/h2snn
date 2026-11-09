// Render the cumulative report to A4 PDF with Playwright (printBackground: true), as the advisor
// report README asks. Uses the installed Chrome through Playwright's `channel` option, so no browser
// download is needed. Run from this folder: node render_pdf.js
const path = require('path');
const PLAYWRIGHT = process.env.PLAYWRIGHT_MODULE ||
  path.join(process.env.APPDATA, 'npm', 'node_modules', '@playwright', 'cli', 'node_modules', 'playwright');
const { chromium } = require(PLAYWRIGHT);

(async () => {
  const html = path.join(__dirname, 'H2S-NN_progress_report_cumulative.html');
  const pdf = path.join(__dirname, 'H2S-NN_progress_report_cumulative.pdf');
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage();
  await page.goto('file:///' + html.replace(/\\/g, '/'));
  await page.pdf({ path: pdf, format: 'A4', printBackground: true, preferCSSPageSize: true });
  await browser.close();
  console.log('wrote', pdf);
})();
