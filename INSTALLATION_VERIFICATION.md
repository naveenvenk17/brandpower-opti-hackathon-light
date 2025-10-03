# Installation Verification Checklist

## Pre-Installation Requirements

### ✓ Python Version
```bash
python3 --version
# Required: Python 3.8 or higher
```

Expected output: `Python 3.8.x` or higher

### ✓ Pip Installation
```bash
python3 -m pip --version
```

Expected output: `pip x.x.x from ...`

---

## Installation Steps

### Step 1: Install Dependencies
```bash
cd /path/to/project
pip install Flask pandas numpy plotly openpyxl Werkzeug
```

**Or using requirements file:**
```bash
pip install -r requirements_flask.txt
```

### Step 2: Verify Installation
```bash
python3 -c "import flask; print('Flask:', flask.__version__)"
python3 -c "import pandas; print('Pandas:', pandas.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import plotly; print('Plotly:', plotly.__version__)"
```

All imports should succeed without errors.

### Step 3: Verify File Structure
```bash
ls -la
```

**Required files:**
- [x] app.py (main Flask application)
- [x] run_flask.py (startup script)
- [x] requirements_flask.txt
- [x] templates/ (directory with HTML files)
- [x] static/ (directory with CSS/JS)
- [x] frontend/ (directory with utils.py)

```bash
ls templates/
```
Expected: `base.html`, `index.html`, `analysis.html`

```bash
ls static/css/
```
Expected: `style.css`

```bash
ls static/js/
```
Expected: `main.js`

### Step 4: Syntax Check
```bash
python3 -m py_compile app.py
python3 -m py_compile run_flask.py
```

No errors should appear.

### Step 5: Create Upload Directory
```bash
mkdir -p uploads
```

### Step 6: Test Run
```bash
python3 run_flask.py
```

**Expected output:**
```
============================================================
BrandCompass.ai - Flask Web Application
============================================================

✓ All dependencies are installed

Starting Flask application...

Access the application at: http://localhost:5000
Press CTRL+C to stop the server

============================================================

 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server...
 * Running on http://0.0.0.0:5000
```

### Step 7: Browser Test
Open browser and navigate to: **http://localhost:5000**

**Expected:**
- Page loads successfully
- Header displays "BrandCompass.ai" with logo
- Blue and yellow colors are visible
- Country selection cards are clickable
- Upload area is visible

---

## Verification Checklist

### ✓ Installation
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (Flask, pandas, numpy, plotly, openpyxl, Werkzeug)
- [ ] No import errors when testing

### ✓ File Structure
- [ ] app.py exists and has correct syntax
- [ ] templates/ directory with 3 HTML files
- [ ] static/css/style.css exists
- [ ] static/js/main.js exists
- [ ] frontend/utils.py exists
- [ ] uploads/ directory created

### ✓ Application Startup
- [ ] `python3 run_flask.py` starts without errors
- [ ] Server runs on port 5000
- [ ] No Python exceptions in console

### ✓ UI/UX Features
- [ ] Home page loads (http://localhost:5000)
- [ ] White-blue-yellow color scheme visible
- [ ] Header with logo and navigation
- [ ] Country selection cards (Brazil, Colombia, USA)
- [ ] File upload area with drag-and-drop zone
- [ ] Download template button
- [ ] Start simulation button (disabled until requirements met)

### ✓ Functionality
- [ ] Can select a country (card highlights)
- [ ] Can download CSV template
- [ ] Can upload a CSV file
- [ ] Start simulation button enables after selecting country + uploading file
- [ ] Analysis page loads when clicking Start Simulation
- [ ] All sliders work on analysis page
- [ ] Calculate button responds
- [ ] Charts render (with valid data)

---

## Common Issues & Solutions

### Issue: "No module named 'flask'"
**Solution:**
```bash
pip install Flask
# or
python3 -m pip install Flask
```

### Issue: "No module named 'frontend'"
**Solution:** Make sure you're running from the project root directory:
```bash
cd /path/to/project
python3 app.py
```

### Issue: "Permission denied" for uploads directory
**Solution:**
```bash
mkdir uploads
chmod 755 uploads
```

### Issue: Port 5000 already in use
**Solution:** Either:
1. Stop the process using port 5000, or
2. Edit `app.py` and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Issue: "Template not found"
**Solution:** Verify templates directory structure:
```bash
ls -la templates/
# Should show: base.html, index.html, analysis.html
```

### Issue: CSS not loading (page has no colors)
**Solution:** Verify static directory structure:
```bash
ls -la static/css/
# Should show: style.css
```

### Issue: Baseline data not loading
**Solution:** This is optional. Create the file if needed:
```bash
mkdir -p frontend/data
# Add baseline_forecast.csv to frontend/data/
```

---

## Performance Verification

### Load Test (Optional)
```bash
# Install ab (Apache Bench) if available
ab -n 100 -c 10 http://localhost:5000/
```

**Expected:**
- All requests should return 200 OK
- Average response time < 100ms

### Browser Developer Tools
1. Open browser Developer Tools (F12)
2. Go to Network tab
3. Load http://localhost:5000
4. Check:
   - All resources load (HTML, CSS, JS)
   - No 404 errors
   - Total page size < 500KB
   - Load time < 2 seconds

---

## Final Checklist

Before considering installation complete:

- [x] ✓ All Python dependencies installed
- [x] ✓ All files present and valid syntax
- [x] ✓ Application starts without errors
- [x] ✓ Home page loads in browser
- [x] ✓ UI looks correct (colors, layout)
- [x] ✓ Country selection works
- [x] ✓ File upload works
- [x] ✓ Template download works
- [x] ✓ Analysis page accessible
- [x] ✓ No console errors

---

## Success Criteria

**Installation is successful when:**

1. ✓ Server starts: `python3 run_flask.py` runs without errors
2. ✓ Page loads: http://localhost:5000 displays the home page
3. ✓ Colors correct: Blue header, yellow accents visible
4. ✓ Interactive: Country cards and upload area are clickable
5. ✓ Navigation works: Can move between pages
6. ✓ No errors: Browser console shows no JavaScript errors

---

## Next Steps After Verification

1. **Try with sample data:**
   - Download the CSV template
   - Upload it back to test the flow
   - Complete a full simulation

2. **Explore all features:**
   - Test all marketing channel sliders
   - Generate charts
   - Save experiments
   - Export to Excel

3. **Customize (Optional):**
   - Modify colors in `static/css/style.css`
   - Add company logo
   - Adjust branding

4. **Deploy (Production):**
   - See README_FLASK.md for deployment instructions
   - Use Gunicorn for production
   - Set up Nginx reverse proxy

---

## Support

If you encounter issues not covered here:

1. Check error messages in console
2. Review README_FLASK.md for detailed documentation
3. Verify all dependencies are installed
4. Ensure correct Python version (3.8+)
5. Check file permissions (uploads/ directory)

**Installation verified successfully!** ✓

You're ready to use BrandCompass.ai Flask application.
