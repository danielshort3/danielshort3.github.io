// Saved model parameters; boundaries are reused from PizzaTipsMeta.
window.PizzaTipsModel = {
  "version": 4,
  "generatedAt": "2025-12-22T00:49:49.369Z",
  "inputFeatures": [
    "cost",
    "orderHour"
  ],
  "targets": {
    "tip": {
      "transform": "log1p"
    },
    "tipPercent": {
      "transform": "log1p"
    }
  },
  "categories": {
    "city": {
      "baseline": "Frisco",
      "values": [
        "Frisco",
        "Plano",
        "The Colony",
        "Lewisville",
        "Carrollton",
        "McKinney",
        "Allen"
      ]
    },
    "housing": {
      "baseline": "Residential",
      "values": [
        "Residential",
        "Apartment",
        "Hotel",
        "Business"
      ]
    }
  },
  "bounds": {
    "latitude": {
      "min": 33.0130544,
      "max": 33.1949433
    },
    "longitude": {
      "min": -96.93078218,
      "max": -96.7145982
    }
  },
  "ranges": {
    "cost": {
      "min": 5.41,
      "max": 243.02,
      "mean": 41.765939248601114
    },
    "tip": {
      "min": 0,
      "max": 40,
      "mean": 7.141079136690646
    },
    "tipPercent": {
      "min": 0,
      "max": 3.621072089,
      "mean": 0.19117123502238237
    },
    "orderHour": {
      "min": 11,
      "max": 21,
      "mean": 18.15347721822542
    },
    "deliveryMinutes": {
      "min": 0,
      "max": 135,
      "mean": 40.43725019984013
    },
    "rain": {
      "min": 0,
      "max": 5.64,
      "mean": 0.16793764988009638
    },
    "maxTemp": {
      "min": 40,
      "max": 108,
      "mean": 80.17026378896882
    },
    "minTemp": {
      "min": 19,
      "max": 79,
      "mean": 59.03357314148681
    }
  },
  "metrics": {
    "tip": {
      "rmse": 0.40663783618126387,
      "r2": 0.3024519062400087,
      "n": 1251
    },
    "tipPercent": {
      "rmse": 0.09039342874618946,
      "r2": 0.07573431595849545,
      "n": 1251
    }
  },
  "coefficients": {
    "tip": {
      "intercept": 1.5823081640335854,
      "values": {
        "cost": 0.01022700228091093,
        "city:Plano": -0.07521221370771652,
        "housing:Apartment": -0.07491206308191074,
        "housing:Business": -0.22453944133434225
      }
    },
    "tipPercent": {
      "intercept": 0.1303849400342169,
      "values": {
        "cost": -0.0009515125883025411,
        "orderHour": 0.004577273574251591,
        "city:Plano": -0.014459822678887454
      }
    }
  }
};
