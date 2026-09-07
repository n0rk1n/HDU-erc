// Run tests.ui.preview_bubbles first. No signed-in browser or external model is used.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 980 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const output = path.resolve('docs/verification/chat-bubbles');
    fs.mkdirSync(output, { recursive: true });
    const user = `bubbles-${Date.now()}`;
    const bodies = () => page.locator('.message--assistant .message-body');
    async function enter(name) {
      await page.locator('#identifier-input').fill(name);
      await page.locator('#enter-chat').click();
      await page.waitForFunction(() => !document.querySelector('#chat-view').hidden && !document.querySelector('#message-input').disabled);
    }
    async function send(text) {
      await page.locator('#message-input').fill(text);
      await page.locator('#send-message').click();
    }
    async function done() {
      await page.waitForFunction(() => !document.querySelector('#message-input').disabled);
    }
    await page.goto('http://127.0.0.1:8766');
    await enter(user);
    await page.evaluate(() => {
      window.bubbleTimes = [];
      const seen = new Set();
      new MutationObserver(() => {
        for (const bubble of document.querySelectorAll('.message--assistant .message-body')) {
          if (bubble.textContent && !seen.has(bubble.textContent)) {
            seen.add(bubble.textContent);
            window.bubbleTimes.push(performance.now());
          }
        }
      }).observe(document.querySelector('#message-list'), { childList: true, subtree: true });
    });
    await send('今天上班好累，事情一直没停下来。');
    await bodies().first().waitFor();
    assert.equal(await bodies().count(), 1, 'first bubble arrives before the rest is generated');
    assert.equal(await page.locator('#message-input').isDisabled(), true);
    await done();
    const expected = ['今天确实够忙的。', '忙完了就先让自己歇一会儿。\n不急着把剩下的事情都安排上。', '要是想聊聊，我在听。'];
    assert.deepEqual(await bodies().allTextContents(), expected);
    const times = await page.evaluate(() => window.bubbleTimes);
    assert.equal(times.length, 3);
    assert.ok(times[2] - times[1] >= 450, 'display queue honors the configured 500 ms gap');
    assert.equal(await page.locator('.message--assistant').count(), 1, 'one turn keeps one record/group');
    assert.equal(await page.locator('.process-card').count(), 1, 'emotion details are shown once per turn');
    await page.screenshot({ path: path.join(output, 'desktop.png'), fullPage: true, animations: 'disabled' });

    await page.reload();
    await enter(user);
    assert.deepEqual(await bodies().allTextContents(), expected, 'reload keeps saved boundaries');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, 'mobile.png'), fullPage: true, animations: 'disabled' });
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    }

    await page.locator('#switch-user').click();
    await enter(user + '-failed');
    await send('模拟后续失败');
    await done();
    assert.deepEqual(await bodies().allTextContents(), ['今天确实够忙的。']);
    assert.equal(await page.locator('.message--assistant').getAttribute('data-state'), 'failed');
    await page.reload();
    await enter(user + '-failed');
    assert.deepEqual(await bodies().allTextContents(), ['今天确实够忙的。']);

    await page.locator('#switch-user').click();
    await enter(user + '-switch');
    await send('检查切换');
    await page.waitForFunction(() => document.querySelectorAll('.message--assistant .message-body').length === 2);
    await page.locator('#switch-user').click();
    await enter(user + '-new');
    // Let any stale display delay expire while observing the new user's empty history.
    await page.waitForTimeout(750);
    assert.equal(await bodies().count(), 0);
    assert.equal(await page.locator('#current-identifier').textContent(), user + '-new');
    assert.equal(await page.locator('#message-input').isDisabled(), false);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ passed: true, scenarios: ['early bubble', '500 ms pacing', 'single emotion card', 'reload', '390/320px', 'partial failure', 'switch during delay'], bubbleIntervalsMs: [times[1] - times[0], times[2] - times[1]] }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
