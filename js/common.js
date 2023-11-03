document.addEventListener("DOMContentLoaded", () => {
  // Insert navigation and social media links
  document.getElementById("combined-header-nav").insertAdjacentHTML('beforeend', `
    <div class="header-container">
     <a href="index.html" class="header-text">
       <div class="header-logo">
        <img src="images/logo.png" alt="Daniel Short Logo" id="logo" />
      </div>
        <h1 class="header-name">aniel Short</h1>
      </a>
      <div class="header-section">
        <div class="nav-links" id="nav-links"> <!-- Added ID for toggling -->
          <a href="index.html">About Me</a>
          <a href="projects.html">Projects</a>
          <a href="contact.html">Contact</a>
          <a href="documents/Resume.pdf" target="_blank" class="resume-link">Resume</a>
        </div>
      </div>
      <div class="social-icons">
        <a href="mailto:danielshort3@gmail.com" target="_blank"><i class="fas fa-envelope"></i></a>
        <a href="https://www.linkedin.com/in/danielshort3/" target="_blank"><i class="fab fa-linkedin-in"></i></a>
        <a href="https://github.com/danielshort3" target="_blank"><i class="fab fa-github"></i></a>
      </div>
    </div>
  `);



  const filterButtons = document.querySelectorAll("#filter-menu button");
  const projectCards = document.querySelectorAll("#projects .project-card");

  projectCards.forEach(card => card.dataset.visible = "true");

  const projectsSection = document.getElementById("projects");
  const navigationPane = document.getElementById("combined-header-nav");

filterButtons.forEach(button => {
  button.addEventListener("click", (event) => {
    
    // Step 1: Remove '.active' from all buttons
    filterButtons.forEach(btn => {
      btn.classList.remove("active");
    });

    // Step 2: Add '.active' to the clicked button
    event.target.classList.add("active");

    const filter = event.target.dataset.filter.split(',');

      // Set initial condition: Add 'fade-out' class
        projectCards.forEach(card => {
          card.classList.add("fade-out");
        });

        // First timeout to handle fade-out
        setTimeout(() => {
          projectCards.forEach(card => {
            const tools = card.querySelector("[data-tools]").dataset.tools.split(',');
            card.dataset.visible = filter.includes("all") || filter.some(f => tools.includes(f)) ? "true" : "false";

            // Remove the fade-out class but wait for some time
            setTimeout(() => {
              card.classList.remove("fade-out");
            }, 20); // a slight delay to allow the browser to recognize the changes

          });

          // Existing code for scroll and align
          const offsetTop = projectsSection.getBoundingClientRect().top + window.pageYOffset;
          const navHeight = navigationPane.offsetHeight;

          // Only execute if window width is 768 pixels or less (mobile view)
          if (window.innerWidth <= 768) {
            window.scrollTo({
              top: offsetTop - navHeight,
              behavior: 'smooth'
            });
          }

        }, 400); // existing 400ms timeout
      });
    });

  // Check for hash and open corresponding modal
  const initialHash = window.location.hash.substr(1);  // Remove the '#'
  if (initialHash && document.getElementById(initialHash)) {
    showModal(initialHash);
  }


  // Modal handling (Test rigorously on mobile)
  const modals = document.querySelectorAll('.modal');
  
  modals.forEach(modal => modal.addEventListener('click', event => {
    if (event.target === modal) {
      closeModal(modal.id);
    }
  }));
});

let currentOpenModal = null;

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  modal.style.display = 'none';
  modal.style.zIndex = '';  // Reset z-index value
  
  // Optionally, re-enable interactions with other elements
  document.body.style.pointerEvents = 'auto';

  currentOpenModal = null;
  
  toggleHoverOnProjectCards('remove');
}

function toggleHoverOnProjectCards(action) {
  const projectCards = document.querySelectorAll('.project-card');
  projectCards.forEach(card => {
    card.classList[action]('no-hover');
  });
}

function showModal(modalId) {
document.body.style.overflow = 'hidden';
 if (currentOpenModal) {
    closeModal(currentOpenModal);
  }

  const modal = document.getElementById(modalId);
  modal.classList.add('show');  // Add 'show' class
  modal.style.display = 'block';
  modal.style.zIndex = 1000;

  // Optionally, disable interactions with other elements
  document.body.style.pointerEvents = 'none';
  modal.style.pointerEvents = 'auto';

  window.location.hash = modalId;
  currentOpenModal = modalId;

  toggleHoverOnProjectCards('add');
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  modal.classList.remove('show');  // Remove 'show' class

  // Use a timeout to match the transition time to hide the modal
  setTimeout(() => {
    modal.style.display = 'none';
    modal.style.zIndex = '';  // Reset z-index value
  }, 500);  // 500ms matches the CSS transition time

  // Optionally, re-enable interactions with other elements
  document.body.style.pointerEvents = 'auto';
  document.body.style.overflow = 'auto';  // Enable background scroll

  // Replace the current history entry without affecting the view
  history.replaceState(null, document.title, window.location.pathname + window.location.search);

  currentOpenModal = null;

  toggleHoverOnProjectCards('remove');
}

let counter = 1;
let intervalId;
let isTransitioning = false;

// Count the number of skill elements
const numberOfSkills = document.querySelectorAll('.skill').length;

// Modify the createNotches function to remove click event listeners
function createNotches() {
  const notchContainer = document.querySelector('.notch-container');
  for (let i = 1; i <= numberOfSkills; i++) {
    const notch = document.createElement('div');
    notch.classList.add('notch');
    notch.id = `notch${i}`;
    // Removed event listener to make notch non-clickable
    notchContainer.appendChild(notch);
  }
}

function updateActiveNotch() {
  // Remove 'active', 'leaving' and 'incoming' classes from all notches
  document.querySelectorAll('.notch').forEach(n => {
    n.classList.remove('active', 'leaving', 'incoming');
  });

  // Add 'active' class to the currently active notch
  document.getElementById(`notch${counter}`).classList.add('active');

  // Calculate the next notch
  const nextCounter = (counter % numberOfSkills) + 1;

  // Add 'incoming' class to the next notch
  document.getElementById(`notch${nextCounter}`).classList.add('incoming');
}

function cycleSkills(direction = 'forward', targetCounter = null) {
  if (isTransitioning) return;

  isTransitioning = true;

  clearInterval(intervalId);
  startInterval();

  const oldSkill = document.querySelector('.skill.active');

  if (oldSkill) {
    oldSkill.classList.remove('active');
    oldSkill.classList.add('leaving');
  }

  let slideFrom = 'none'; // By default, do not slide from any direction
  if (targetCounter !== null) {
    slideFrom = targetCounter > counter ? 'right' : 'left';
    counter = targetCounter;
  } else {
    if (direction === 'forward') {
      counter = (counter % numberOfSkills) + 1;
      slideFrom = 'right';
    } else {
      counter = ((counter - 2 + numberOfSkills) % numberOfSkills) + 1;
      slideFrom = 'left';
    }
  }

  const newSkill = document.getElementById(`skill${counter}`);
  
  newSkill.classList.remove('leaving');
  newSkill.classList.add('active');
  
  // Temporarily disable transitions to set the initial position
  newSkill.style.transition = 'none';
  oldSkill.style.transition = 'none';
  
  // Set the starting position
  newSkill.style.transform = (slideFrom === 'right') ? 'translateX(100%)' : 'translateX(-100%)';
  oldSkill.style.transform = 'translateX(0)';
  
  // Force a DOM reflow
  newSkill.getBoundingClientRect();
  oldSkill.getBoundingClientRect();
  
  // Re-enable the transitions
  newSkill.style.transition = 'transform 1s ease-in-out';
  oldSkill.style.transition = 'transform 1s ease-in-out';
  
  // Animate to the final position
  newSkill.style.transform = 'translateX(0)';
  oldSkill.style.transform = (direction === 'forward') ? 'translateX(-100%)' : 'translateX(100%)';

  updateActiveNotch();

  // Reset after transition
  newSkill.addEventListener('transitionend', function() {
    isTransitioning = false;
    oldSkill.classList.remove('leaving');
    oldSkill.style.transition = 'none';
  }, { once: true });
}




// Initialize the notches and first active notch
createNotches();
updateActiveNotch();

// Initialize the first skill as active
document.getElementById(`skill${counter}`).classList.add('active');

// Function to start the interval
function startInterval() {
  intervalId = setInterval(function() {
    cycleSkills('forward');
  }, 3000); // Interval time
}

// Start the interval initially
startInterval();

// Listen for visibility changes
document.addEventListener('visibilitychange', function() {
  if (document.hidden) {
    clearInterval(intervalId);
  } else {
    startInterval();
  }
});

// Add event listeners for buttons
document.getElementById('prevBtn').addEventListener('click', function() {
  if (isTransitioning) return; // If a transition is ongoing, do nothing
  cycleSkills('backward');
});

document.getElementById('nextBtn').addEventListener('click', function() {
  if (isTransitioning) return; // If a transition is ongoing, do nothing
  cycleSkills('forward');
});

