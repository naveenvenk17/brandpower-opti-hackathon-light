# BrandCompass.ai - Quick Start Guide

## Installation (3 Steps)

### 1. Install Dependencies
```bash
pip install Flask pandas numpy plotly openpyxl Werkzeug
```

Or use the requirements file:
```bash
pip install -r requirements_flask.txt
```

### 2. Run the Application
```bash
python3 run_flask.py
```

Or directly:
```bash
python3 app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

---

## Using the Application (5 Steps)

### Step 1: Select Country 🌍
Click on your target market:
- 🇧🇷 Brazil
- 🇨🇴 Colombia
- 🇺🇸 USA

### Step 2: Upload Data 📊
- Click "Choose a CSV file" or drag and drop
- Or download the template first to see the format
- Required columns: Brand, Year, Month, Week, marketing channels

### Step 3: Start Simulation 🚀
- Click "Start Simulation" button
- You'll be taken to the analysis page

### Step 4: Adjust & Calculate 🎛️
- Select brands, years, and months to analyze
- Adjust marketing spend sliders (-50% to +50%)
- Click "Calculate Brand Power"
- View results in table and chart

### Step 5: Save Experiment 💾
- Enter an experiment name
- Click "Save Experiment"
- Export all experiments to Excel anytime

---

## File Structure

```
project/
├── app.py              ← Main Flask application
├── run_flask.py        ← Easy startup script
├── requirements_flask.txt
│
├── templates/          ← HTML pages
│   ├── base.html
│   ├── index.html
│   └── analysis.html
│
├── static/            ← CSS & JavaScript
│   ├── css/style.css
│   └── js/main.js
│
├── frontend/          ← Business logic
│   └── utils.py
│
└── uploads/           ← User files (auto-created)
```

---

## Troubleshooting

### "Module not found" Error
```bash
pip install Flask pandas numpy plotly openpyxl Werkzeug
```

### Port 5000 Already in Use
Edit `app.py`, change last line to:
```python
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Upload Directory Error
```bash
mkdir uploads
```

### Baseline Data Not Loading
Create the file: `frontend/data/baseline_forecast.csv`
(Optional - application works without it)

---

## Key Features

✓ **Country Selection** - Brazil, Colombia, USA
✓ **CSV Upload** - Drag & drop support
✓ **Template Download** - Sample data format
✓ **Multi-Brand Analysis** - Compare multiple brands
✓ **Marketing Optimization** - 16 channels
✓ **Quarterly Forecasts** - Q3 2024 - Q2 2025
✓ **Interactive Charts** - Plotly visualizations
✓ **Experiment Saving** - Save & compare scenarios
✓ **Excel Export** - Download all experiments
✓ **Mobile Responsive** - Works on all devices

---

## Color Scheme

Professional white-blue-yellow palette:
- Primary: Blue (#1e50a2)
- Accent: Yellow (#ffc107)
- Background: White/Off-white
- Text: Dark gray (#212529)

---

## Browser Support

- ✓ Chrome/Edge 90+
- ✓ Firefox 88+
- ✓ Safari 14+
- ✓ Mobile browsers

---

## Production Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY=your-secret-key-here
```

### With Nginx
See README_FLASK.md for detailed Nginx configuration

---

## Support Files

- **README_FLASK.md** - Complete documentation
- **CONVERSION_SUMMARY.md** - Migration details
- **requirements_flask.txt** - Python dependencies

---

## Data Format Example

```csv
Brand,Year,Month,Week,brand_events,brand_promotion,...
BrandA,2024,7,1,100,200,...
BrandA,2024,7,2,110,210,...
BrandB,2024,7,1,80,180,...
```

**Required Columns:**
- Brand (or brand)
- Year (or year)
- Month (or month)
- Week (or week, week_of_month)

**Marketing Channels (16):**
brand_events, brand_promotion, digitaldisplayandsearch, digitalvideo, influencer, meta, ooh, opentv, others, paytv, radio, sponsorship, streamingaudio, tiktok, twitter, youtube

---

## Quick Commands

```bash
# Install dependencies
pip install -r requirements_flask.txt

# Run development server
python3 run_flask.py

# Run with Gunicorn (production)
gunicorn -w 4 app:app

# Check syntax
python3 -m py_compile app.py

# Create uploads directory
mkdir -p uploads frontend/data
```

---

## Next Steps

1. ✓ Install dependencies
2. ✓ Run the application
3. ✓ Open http://localhost:5000
4. Upload your data or try the template
5. Explore the analysis features
6. Save and compare experiments
7. Export results to Excel

---

**Need Help?**
- Check README_FLASK.md for detailed documentation
- Review CONVERSION_SUMMARY.md for technical details
- Verify all dependencies are installed

**Enjoy using BrandCompass.ai!** 🧭
