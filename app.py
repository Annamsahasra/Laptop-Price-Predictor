from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import re
import os

app = Flask(__name__)

# Load model and feature names from files
try:
    model = joblib.load("LaptopPriceModel.pkl")
    feature_names = joblib.load("FeatureNames.pkl")
    # Note: Scaler.pkl is loaded but not used because XGBRegressor does not use scaled features.
    # Scaler was only used for the baseline Linear Regression model in the notebook.
    if os.path.exists("Scaler.pkl"):
        scaler = joblib.load("Scaler.pkl")
    else:
        scaler = None
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    feature_names = []
    scaler = None


def sanitize_column_name(col_name):
    """Sanitizes feature column names to match the preprocessing in Cell 40 of the notebook."""
    return (col_name
            .replace('[', '')
            .replace(']', '')
            .replace('<', '')
            .replace(' ', '_')
            .replace('=', '_')
            .replace('/', '_')
            .replace('(', '')
            .replace(')', '')
            .replace('&', '')
            .replace(':', '')
            .replace(',', ''))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not feature_names:
        return render_template(
            "index.html",
            error="Machine Learning model files could not be loaded on the server. Please check the logs."
        )

    try:
        # 1. Retrieve and Validate Inputs
        company = request.form.get("company", "").strip()
        typename = request.form.get("typename", "").strip()
        cpu_brand = request.form.get("cpu_brand", "").strip()
        gpu_brand = request.form.get("gpu_brand", "").strip()
        os_name = request.form.get("os", "").strip()

        # Check for missing categorical selections
        if not all([company, typename, cpu_brand, gpu_brand, os_name]):
            raise ValueError("All categorical dropdown selections are required.")

        # Numeric validations
        ram = request.form.get("ram")
        weight = request.form.get("weight")
        inches = request.form.get("inches")
        resolution = request.form.get("resolution", "").strip()

        if not all([ram, weight, inches, resolution]):
            raise ValueError("All specifications including RAM, Weight, Screen Size, and Resolution are required.")

        try:
            ram = int(ram)
            if ram <= 0:
                raise ValueError("RAM must be a positive integer.")
        except ValueError:
            raise ValueError("Invalid RAM format. Must be an integer greater than zero.")

        try:
            weight = float(weight)
            if weight <= 0:
                raise ValueError("Weight must be a positive number.")
        except ValueError:
            raise ValueError("Invalid Weight format. Must be a decimal number greater than zero.")

        try:
            inches = float(inches)
            if inches <= 0:
                raise ValueError("Screen size (inches) must be a positive number.")
        except ValueError:
            raise ValueError("Invalid Screen Size format. Must be a decimal number greater than zero.")

        # Resolution validation & parsing
        res_match = re.search(r'(\d+)x(\d+)', resolution)
        if not res_match:
            raise ValueError("Invalid screen resolution format. Use standard like '1920x1080'.")
        
        x_res = int(res_match.group(1))
        y_res = int(res_match.group(2))
        if x_res <= 0 or y_res <= 0:
            raise ValueError("Resolution width and height must be greater than zero.")

        # Storage parsing
        ssd = int(request.form.get("ssd", 0))
        hdd = int(request.form.get("hdd", 0))
        flash = int(request.form.get("flash", 0))
        hybrid = int(request.form.get("hybrid", 0))

        if any(v < 0 for v in [ssd, hdd, flash, hybrid]):
            raise ValueError("Storage capacity cannot be negative.")

        # 2. Feature Engineering
        # Calculate PPI
        ppi = (np.sqrt(x_res ** 2 + y_res ** 2)) / inches
        
        # Calculate Total Storage
        total_storage = ssd + hdd + flash + hybrid

        # Touchscreen and IPS Panel indicators
        touchscreen = 1 if request.form.get("touchscreen") == "1" else 0
        ips = 1 if request.form.get("ips") == "1" else 0

        # 3. Recreate Feature Vector
        # Create a single row DataFrame initialized with all 0s
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)

        # Populate numerical features
        input_df['Ram_GB'] = ram
        input_df['Weight_Kg'] = weight
        input_df['SSD'] = ssd
        input_df['HDD'] = hdd
        input_df['Flash_Storage'] = flash
        input_df['Hybrid'] = hybrid
        input_df['Total_Storage'] = total_storage
        input_df['Touchscreen'] = touchscreen
        input_df['IPS'] = ips
        input_df['PPI'] = ppi

        # Set categorical dummy columns to 1 (if they exist in feature_names)
        company_col = sanitize_column_name(f"Company_{company}")
        typename_col = sanitize_column_name(f"TypeName_{typename}")
        cpu_col = sanitize_column_name(f"CPU_Brand_{cpu_brand}")
        gpu_col = sanitize_column_name(f"GPU_Brand_{gpu_brand}")
        os_col = sanitize_column_name(f"OS_{os_name}")

        feature_names_set = set(feature_names)
        for dummy_col in [company_col, typename_col, cpu_col, gpu_col, os_col]:
            if dummy_col in feature_names_set:
                input_df.loc[0, dummy_col] = 1

        # Reorder columns to match FeatureNames.pkl exactly
        input_df = input_df[feature_names]

        # 4. Predict Price
        predicted_price = model.predict(input_df)[0]
        
        # Clip negative prices (if any random/unusual configs lead to negative predictions)
        predicted_price = max(0.0, float(predicted_price))

        # Format storage description for display
        storage_parts = []
        if ssd > 0:
            storage_parts.append(f"{ssd}GB SSD")
        if hdd > 0:
            storage_parts.append(f"{hdd}GB HDD")
        if flash > 0:
            storage_parts.append(f"{flash}GB Flash")
        if hybrid > 0:
            storage_parts.append(f"{hybrid}GB Hybrid")
        
        storage_desc = " + ".join(storage_parts) if storage_parts else "No Storage"

        specs_summary = {
            "company": company,
            "typename": typename,
            "cpu_brand": cpu_brand,
            "gpu_brand": gpu_brand,
            "ram": ram,
            "os": os_name,
            "weight": weight,
            "inches": inches,
            "resolution": resolution,
            "touchscreen": touchscreen,
            "ips": ips,
            "ppi": ppi,
            "ssd": ssd,
            "hdd": hdd,
            "flash": flash,
            "hybrid": hybrid,
            "total_storage": total_storage,
            "storage_desc": storage_desc
        }

        return render_template("result.html", price=predicted_price, specs=specs_summary)

    except ValueError as val_err:
        return render_template("index.html", error=str(val_err))
    except Exception as err:
        return render_template("index.html", error=f"An unexpected error occurred during prediction: {str(err)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)