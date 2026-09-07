// Start tests.ui.preview_emotion first. Set PLAYWRIGHT_MODULE to a local package if needed.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const output = path.resolve('docs/verification/emotion-ui');
    fs.mkdirSync(output, { recursive: true });
    const user = `ui-qa-${Date.now()}`;
    async function enter(name) {
      await page.locator('#identifier-input').fill(name);
      await page.locator('#enter-chat').click();
      await page.locator('#message-input').waitFor({ state: 'visible' });
      await page.waitForFunction(() => !document.querySelector('#message-input').disabled);
    }
    async function send(text) {
      await page.locator('#message-input').fill(text);
      await page.locator('#send-message').click();
    }
    async function completed() {
      await page.waitForFunction(() => !document.querySelector('#message-input').disabled);
    }
    await page.goto('http://127.0.0.1:8765');
    await enter(user);
    assert.equal(await page.locator('#emotion-label').innerText(), '尚未识别');
    await send('明天要做汇报，我很担心讲不好，想到这里就有些紧张。');
    await page.locator('.process-step[data-state="running"] .step-title', { hasText: '分析对话情绪' }).waitFor();
    assert.equal(await page.locator('.process-card').evaluate(el => el.open), true);
    assert.equal(await page.locator('#emotion-label').innerText(), '尚未识别');
    await page.screenshot({ path: path.join(output, 'desktop-processing.png'), fullPage: true, animations: "disabled" });
    await completed();
    const firstCard = page.locator('.process-card').first();
    assert.equal(await firstCard.evaluate(el => el.open), false);
    assert.equal(await page.locator('#emotion-label').innerText(), '焦虑');
    await firstCard.locator('summary').click();
    assert.match(await firstCard.innerText(), /模型置信度 88%/);
    assert.equal(await firstCard.locator('.process-step[data-state="completed"]').count(), 4);
    await page.screenshot({ path: path.join(output, 'desktop-result.png'), fullPage: true, animations: "disabled" });
    await page.reload();
    await enter(user);
    assert.equal(await page.locator('#emotion-label').innerText(), '焦虑');
    await send('谢谢，我们先聊聊怎么开场吧。');
    await completed();
    const skipped = page.locator('.process-card').last();
    await skipped.locator('summary').click();
    assert.match(await skipped.innerText(), /本轮未调用情绪识别/);
    assert.equal(await skipped.locator('.emotion-result').count(), 0);
    assert.equal(await page.locator('#emotion-label').innerText(), '焦虑');
    await send('识别失败');
    await completed();
    const failed = page.locator('.process-card').last();
    await failed.locator('summary').click();
    assert.match(await failed.innerText(), /本次识别未完成/);
    assert.equal(await failed.locator('.emotion-result').count(), 0);
    assert.equal(await failed.locator('.process-step[data-state="failed"]').count(), 1);
    assert.equal(await page.locator('#emotion-label').innerText(), '焦虑');
    await page.locator('#switch-user').click();
    await enter(`${user}-mobile`);
    assert.equal(await page.locator('#emotion-label').innerText(), '尚未识别');
    await page.setViewportSize({ width: 390, height: 844 });
    await send('明天要汇报，我担心讲不好，有些紧张。');
    await completed();
    await page.locator('.process-card summary').click();
    await page.locator('.emotion-result').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, 'mobile-result.png'), fullPage: true, animations: "disabled" });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    const stage = await page.locator('.conversation-stage').boundingBox();
    const composer = await page.locator('.composer-area').boundingBox();
    assert(stage.y + stage.height <= composer.y + 1, 'composer must not cover the scrollable conversation');
    const badge = await page.locator('#emotion-badge').boundingBox();
    assert(badge.x >= 0 && badge.x + badge.width <= 390);
    await page.setViewportSize({ width: 320, height: 700 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    // Leave an accepted turn running, switch users and reject any late UI updates.
    await page.locator('#switch-user').click();
    await enter(`${user}-late`);
    await send('明天汇报很紧张');
    await page.locator('.process-card').waitFor();
    await page.locator('#switch-user').click();
    await enter(`${user}-empty`);
    await page.waitForTimeout(4500);
    assert.equal(await page.locator('#emotion-label').innerText(), '尚未识别');
    assert.equal(await page.locator('.message').count(), 0);
    // The detached turn continued and is reconstructed from real persisted facts.
    await page.locator('#switch-user').click();
    await enter(`${user}-late`);
    assert.equal(await page.locator('#emotion-label').innerText(), '焦虑');
    assert.match(await page.locator('.process-card').innerText(), /88%/);
    assert.deepEqual(errors, []);
    console.log('Browser QA passed: live progress, success, skip, failure, reload, mobile, user isolation, detached recovery.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
