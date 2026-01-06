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
    - **CSS**: Modify styles in `css/styles.css` to change the appearance.
    - **JavaScript**: Update functionality in `js/common.js`.

## Short links (private dashboard)

This repo includes a small, Bitly-style redirect service intended for a single trusted admin (you).

- **Redirects:** `https://<your-domain>/go/<slug>` returns a `301` or `302` to the stored destination.
- **Dashboard:** `pages/short-links.html` (enter your admin token to list/create/update/delete links).
- **Admin auth:** set `SHORTLINKS_ADMIN_TOKEN` in your Vercel environment (do not commit it).
- **Storage:** AWS DynamoDB (set `SHORTLINKS_DDB_TABLE`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).

Generate an admin token locally:

- `openssl rand -hex 32`
- `node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`

## File Structure

```plaintext
.
├── css
│   └── styles.css           # CSS styles for the website
├── img                   # Folder for all images used in the website
│   ├── logo.png
│   ├── head.jpg
│   ├── project_1.png
│   ├── project_2.png
│   ├── ...
├── js
│   ├── common.js                 # Site-wide utilities
│   ├── projects-data.js          # Portfolio project definitions
│   ├── portfolio.js              # Portfolio UI logic
│   ├── contributions-data.js     # Public contributions definitions
│   ├── contributions.js          # Build contributions list
│   ├── contributions-carousel.js # Carousel helper
│   └── ga4-events.js             # Google Analytics events
├── portfolio.html           # Projects page
├── contact.html             # Contact page
├── documents                # Folder for downloadable project files
│   ├── Resume.pdf
│   ├── Project_1.xlsx
│   ├── ...
└── README.md                # This readme file
```

## Shape Classifier Demo

The interactive demo calls an AWS Lambda function for real-time predictions. Ensure your Lambda code includes CORS headers so the browser can access it. See [documents/lambda-cors.md](documents/lambda-cors.md) for a minimal example.
