// Simple test runner used by `npm test`
const fs = require('fs');

// Utility to assert that a file includes a specific string
function checkFileContains(file, text) {
  if (!fs.existsSync(file)) {
    throw new Error(`${file} does not exist`);
  }
  const content = fs.readFileSync(file, 'utf8');
  // Simple substring search to keep tests lightweight
  if (!content.includes(text)) {
    throw new Error(`${file} missing expected text: ${text}`);
  }
}

// Run a couple of smoke tests
try {
  checkFileContains('index.html', 'Turning data into actionable insights');
  checkFileContains('contact.html', '<title>Contact │ Daniel Short');
  console.log('All tests passed.');
} catch (err) {
  // Fail the process with a helpful message
  console.error(err.message);
  process.exit(1);
}
