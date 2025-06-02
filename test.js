const fs = require('fs');

function checkFileContains(file, text) {
  if (!fs.existsSync(file)) {
    throw new Error(`${file} does not exist`);
  }
  const content = fs.readFileSync(file, 'utf8');
  if (!content.includes(text)) {
    throw new Error(`${file} missing expected text: ${text}`);
  }
}

try {
  checkFileContains('index.html', 'Turning data into actionable insights');
  checkFileContains('contact.html', '<title>Contact │ Daniel Short');
  console.log('All tests passed.');
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
