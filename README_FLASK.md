# BrandCompass.ai - Flask Web Application

A modern, responsive web application for brand power simulation and analysis, converted from Streamlit to Flask with an enhanced UI featuring a white-blue-yellow color palette.

## Features

### Core Functionality
- **Country Selection**: Choose target market (Brazil, Colombia, USA)
- **Data Upload**: Upload CSV files with brand marketing data
- **Template Download**: Get sample CSV template for data format
- **Brand Power Analysis**: Simulate brand power across multiple quarters
- **Marketing Channel Optimization**: Adjust marketing spend across 16+ channels
- **Interactive Charts**: Visualize baseline vs simulated forecasts
- **Experiment Management**: Save and compare multiple scenarios
- **Excel Export**: Export all experiments to Excel format

### UI/UX Enhancements
- **Modern Design**: Clean, professional interface with white-blue-yellow palette
- **Responsive Layout**: Mobile-friendly design that works on all devices
- **Smooth Animations**: Fade-in effects and smooth transitions
- **Interactive Elements**: Hover effects, drag-and-drop file upload
- **Real-time Feedback**: Live updates and status notifications
- **Intuitive Navigation**: Clear flow from data upload to analysis

## Color Palette

The application uses a professional white-blue-yellow color scheme:

- **Primary Blue**: `#1e50a2` (headers, buttons, accents)
- **Blue Dark**: `#153d7a` (gradients, footer)
- **Blue Light**: `#4a7bc8` (highlights, borders)
- **Accent Yellow**: `#ffc107` (call-to-action, highlights)
- **Yellow Dark**: `#e0a800` (button hover states)
- **White/Off-white**: `#ffffff`, `#f8f9fa` (backgrounds)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements_flask.txt
   ```

2. **Create necessary directories** (if not exists):
   ```bash
   mkdir -p uploads frontend/data
   ```

3. **Prepare baseline data** (optional):
   Place `baseline_forecast.csv` in `frontend/data/` directory

## Running the Application

### Development Mode

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Application Structure

```
project/
├── app.py                      # Main Flask application
├── requirements_flask.txt      # Python dependencies
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Home page
│   └── analysis.html          # Analysis page
├── static/                     # Static assets
│   ├── css/
│   │   └── style.css          # Main stylesheet
│   └── js/
│       └── main.js            # JavaScript functionality
├── frontend/                   # Backend logic
│   ├── utils.py               # Utility functions
│   └── data/                  # Data files
│       └── baseline_forecast.csv
└── uploads/                    # User uploaded files
```

## Usage Guide

### 1. Home Page - Data Setup

**Select Country**:
- Click on Brazil, Colombia, or USA card
- Selected country will be highlighted with blue gradient

**Upload Data**:
- Click "Choose a CSV file" or drag and drop
- Supported format: CSV with brand marketing data
- Required columns: Brand, Year, Month, Week
- Marketing channels: brand_events, brand_promotion, digitalvideo, meta, etc.

**Download Template**:
- Click "Download CSV Template" for sample format
- Template includes all required columns and sample data

**Start Simulation**:
- Both country and data file must be selected
- Click "Start Simulation" to proceed to analysis

### 2. Analysis Page - Brand Power Simulation

**Analysis Parameters**:
- **Select Brands**: Multi-select brands for analysis
- **Select Years**: Choose year range
- **Select Months**: Filter specific months

**Data Preview**:
- View uploaded data in table format
- Toggle "Show as Percentage" for percentage view
- Table shows all optimizable marketing features

**Adjust Marketing Spend**:
- Use sliders to adjust each marketing channel (-50% to +50%)
- Values update in real-time
- Click "Reset All Channels" to return to baseline

**Calculate Brand Power**:
- Click "Calculate Brand Power" button
- System simulates quarterly forecasts (Q3 2024 - Q2 2025)
- Results show baseline vs simulated comparison

**Results Display**:
- **Table**: Shows quarterly power values with change indicators
  - ↗ = Increase vs baseline (green)
  - ↘ = Decrease vs baseline (red)
  - → = Unchanged (<1% change) (gray)
- **Chart**: Interactive Plotly chart with baseline and simulated lines

**Save Experiment**:
- Enter experiment name
- Click "Save Experiment"
- Maximum 5 experiments can be saved
- Export all experiments to Excel

## Data Format

### Input CSV Format

```csv
Brand,Year,Month,Week,brand_events,brand_promotion,digitaldisplayandsearch,...
BrandA,2024,7,1,100,200,150,...
BrandA,2024,7,2,110,210,160,...
BrandB,2024,7,1,80,180,140,...
```

**Required Columns**:
- `Brand` (or `brand`): Brand identifier
- `Year` (or `year`): Year (2024, 2025)
- `Month` (or `month`): Month (1-12)
- `Week` (or `week`, `week_of_month`): Week identifier

**Optimizable Features** (marketing channels):
- brand_events
- brand_promotion
- digitaldisplayandsearch
- digitalvideo
- influencer
- meta
- ooh (out of home)
- opentv
- others
- paytv
- radio
- sponsorship
- streamingaudio
- tiktok
- twitter
- youtube

### Baseline Forecast Format

```csv
year,period,period_type,country,brand,power
2024,Q3,quarterly,Brazil,BrandA,15.5
2024,Q4,quarterly,Brazil,BrandA,16.2
```

## API Endpoints

### POST `/select_country`
Select target country for analysis
- Form data: `country` (Brazil, Colombia, US)

### POST `/upload`
Upload CSV data file
- Multipart form data with file
- Returns: JSON with file info or error

### GET `/download_template`
Download CSV template
- Returns: CSV file download

### GET `/analysis`
Display analysis page
- Requires uploaded data in session
- Returns: HTML page with analysis tools

### POST `/calculate`
Calculate brand power simulation
- JSON body: `{ changes: { channel: percentage } }`
- Returns: JSON with baseline and simulated data

### POST `/save_experiment`
Save experiment to session
- JSON body: `{ name, baseline_data, simulated_data, changes }`
- Returns: JSON success response

### GET `/experiments`
List saved experiments
- Returns: JSON array of experiments

### GET `/export_experiments`
Export experiments to Excel
- Returns: Excel file download

## Technical Details

### Technologies Used
- **Backend**: Flask 3.0+, Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly.js
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Styling**: Custom CSS with CSS Variables

### Browser Compatibility
- Chrome/Edge: 90+
- Firefox: 88+
- Safari: 14+
- Mobile browsers: iOS Safari 14+, Chrome Mobile 90+

### Performance
- Lightweight: No heavy frontend framework
- Fast loading: Optimized CSS and minimal JavaScript
- Responsive: Smooth animations with CSS transforms
- Efficient: Server-side processing with Flask

## Customization

### Changing Colors

Edit `/static/css/style.css`:

```css
:root {
    --primary-blue: #1e50a2;      /* Your primary color */
    --accent-yellow: #ffc107;      /* Your accent color */
    /* ... other variables */
}
```

### Adding New Features

1. **Add route** in `app.py`:
   ```python
   @app.route('/new_feature')
   def new_feature():
       return render_template('new_feature.html')
   ```

2. **Create template** in `templates/`:
   ```html
   {% extends "base.html" %}
   {% block content %}
   <!-- Your content -->
   {% endblock %}
   ```

3. **Update navigation** in `base.html`

## Troubleshooting

### Issue: Application won't start
- Check Python version: `python --version` (3.8+)
- Install dependencies: `pip install -r requirements_flask.txt`
- Check port 5000 is available

### Issue: File upload fails
- Check file is CSV format
- Verify file size < 16MB
- Ensure `uploads/` directory exists and is writable

### Issue: Baseline data not loading
- Check `frontend/data/baseline_forecast.csv` exists
- Verify CSV format matches expected structure
- Check file permissions

### Issue: Charts not displaying
- Ensure Plotly CDN is accessible
- Check browser console for JavaScript errors
- Verify calculation completed successfully

## Differences from Streamlit Version

### Advantages of Flask Version
1. **Better Performance**: Faster page loads, no WebSocket overhead
2. **Custom UI**: Full control over design and user experience
3. **Scalability**: Easier to deploy and scale horizontally
4. **SEO Friendly**: Standard HTML pages vs Streamlit's SPA
5. **Professional Look**: Modern, responsive design with custom branding

### Preserved Features
- All data processing functionality
- Brand power calculation logic
- Multiple brand/year/month selection
- Marketing channel optimization
- Experiment saving and comparison
- Excel export capability

### Enhanced Features
- Drag-and-drop file upload
- Smooth animations and transitions
- Real-time slider feedback
- Interactive charts with Plotly
- Mobile-responsive design
- Professional color scheme

## License

Copyright 2024 BrandCompass.ai. All rights reserved.

## Support

For issues or questions, please contact the development team.
