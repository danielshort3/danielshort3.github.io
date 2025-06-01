# Daniel Short's Portfolio Website

## Overview

Welcome to the GitHub repository for my personal portfolio website. This site showcases my data analysis and machine learning projects, credentials, areas of interest, and provides ways to contact me for collaboration or hiring.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Features

- **Responsive Design**: Ensures the website looks good on all devices.
- **Portfolio Showcase**: Highlights my key projects with detailed descriptions and download links.
- **Modal Pop-ups**: Provides detailed information about each project.
- **Filterable Projects**: Allows users to filter projects by the tools used.
- **Credentials Carousel**: Displays my certifications and credentials with a scrolling feature.
- **Interactive Navigation**: Includes smooth scrolling and dynamic content loading.
- **Social Media Integration**: Links to my email, LinkedIn, and GitHub profiles.

## Technologies Used

- HTML5
- CSS3
- JavaScript (ES6)
- FontAwesome for icons
- Google Fonts (Poppins)

## Setup and Installation

1. **Clone the Repository**: 
    ```bash
    git clone https://github.com/danielshort3/danielshort3.github.io.git
    ```
2. **Navigate to the Directory**:
    ```bash
    cd danielshort3.github.io
    ```
3. **Open the Project**:
    - Open `index.html` in your preferred web browser to view the homepage.

## Usage

- **Navigating the Website**:
    - Click on the navigation links to move between sections.
    - Click on project cards to view detailed information in modal pop-ups.
    - Use the filter buttons to sort projects by the tools used.
    - Navigate through credentials using the carousel buttons.

- **Modifying the Website**:
    - **HTML**: Update content in the `.html` files as needed.
    - **CSS**: Modify styles in the `css` modules to change the appearance.
    - **JavaScript**: Update functionality in the feature modules under `js`.

## File Structure

```plaintext
.
├── css
│   ├── styles.css                    # Aggregates modular CSS
│   ├── variables.css                 # Design tokens
│   ├── base/                         # Base styles
│   │   └── base.css
│   ├── layout/                       # Layout rules
│   │   └── layout.css
│   ├── components/                   # UI components
│   │   ├── components1.css
│   │   └── components2.css
│   └── utilities/                    # Helper classes
│       └── utilities.css
├── js
│   ├── analytics/                    # GA4 events
│   │   └── ga4-events.js
│   ├── utils/                        # Shared helpers
│   │   └── common.js
│   ├── portfolio/                    # Portfolio feature
│   │   ├── projects-data.js
│   │   └── portfolio.js
│   ├── contributions/                # Contributions feature
│   │   ├── contributions.js
│   │   └── contributions-carousel.js
│   └── forms/                        # Forms and modals
│       └── contact.js
├── portfolio.html                # Projects page
├── contact.html                  # Contact page
├── documents                     # Downloadable files
│   ├── Resume.pdf
│   ├── Project_1.xlsx
│   ├── ...
└── README.md                     # This readme file
```
