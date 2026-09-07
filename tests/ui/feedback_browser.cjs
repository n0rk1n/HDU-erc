// Uses the real local app with tests.ui.preview_bubbles (offline models).
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 980}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const name = `feedback-${Date.now()}`;
    const controls = () => page.locator('.reply-feedback');
    const like = () => controls().last().getByRole('button', {name: '点赞这条回复'});
    const dislike = () => controls().last().getByRole('button', {name: '点踩这条回复'});
    async function enter(user) {
      await page.locator('#identifier-input').fill(user);
      await page.locator('#enter-chat').click();
      await page.waitForFunction(() => !document.querySelector('#chat-view').hidden && !document.querySelector('#message-input').disabled);
    }
    async function send(text) {
      await page.locator('#message-input').fill(text);
      await page.locator('#send-message').click();
      await page.waitForFunction(() => !document.querySelector('#message-input').disabled);
    }
    await page.goto('http://127.0.0.1:8766');
    await enter(name);
    await send('今天有点累');
    assert.equal(await controls().count(), 1);
    assert.equal(await page.locator('.message--assistant .message-body').count(), 3);
    assert.equal(await page.locator('.message--user .reply-feedback').count(), 0);
    await page.route('**/messages/*/feedback', route => route.fulfill({status: 500, contentType: 'application/json', body: '{"error":{"code":"database_error"}}'}));
    assert.equal(await like().locator('svg').count(), 1);
    assert.equal(await dislike().locator('svg').count(), 1);
    assert.equal(await like().innerText(), '');
    await like().click();
    await page.getByText('评价保存失败，请重试。').waitFor();
    assert.equal(await like().isEnabled(), true);
    await page.unroute('**/messages/*/feedback');
    await like().click();
    await page.getByText('已赞', {exact: true}).waitFor();
    assert.equal(await controls().last().getByRole('button').count(), 0);
    assert.equal(await controls().last().innerText(), '已赞');
    await page.reload();
    await enter(name);
    await page.getByText('已赞', {exact: true}).waitFor();
    assert.equal(await controls().last().getByRole('button').count(), 0);
    await send('再聊聊');
    await dislike().click();
    await page.getByText('已踩', {exact: true}).waitFor();
    assert.equal(await controls().last().getByRole('button').count(), 0);
    assert.equal(await controls().last().innerText(), '已踩');
    fs.mkdirSync('docs/verification/reply-feedback', {recursive: true});
    await page.screenshot({path: 'docs/verification/reply-feedback/desktop.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    await controls().last().scrollIntoViewIfNeeded();
    await page.screenshot({path: 'docs/verification/reply-feedback/mobile.png', fullPage: true});
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.locator('#switch-user').click();
    await enter(name + '-other');
    assert.equal(await controls().count(), 0);
    await send('模拟失败');
    assert.equal(await controls().count(), 0, 'failed reply cannot be rated');
    assert.deepEqual(errors, []);
    console.log('feedback browser scenarios passed: like, dislike, failure retry, reload, per-reply controls, mobile, user isolation, failed reply');
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exitCode = 1;});
