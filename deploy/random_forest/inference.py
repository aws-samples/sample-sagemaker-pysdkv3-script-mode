import joblib
import numpy as np
import os
import json
from djl_python import Input, Output

_model = None


def handle(inputs: Input) -> Output:
    """
    DJL Serving handler for scikit-learn model.
    """
    global _model

    if inputs.is_empty():
        properties = inputs.get_properties()
        model_dir = properties.get("model_dir", "/opt/ml/model")
        _model = joblib.load(os.path.join(model_dir, "model.joblib"))
        return None

    content_type = inputs.get_property("Content-Type") or "text/csv"
    request_body = inputs.get_as_string()

    if content_type == "text/csv":
        rows = request_body.strip().split("\n")
        data = []
        for row in rows:
            if row:
                data.append([float(v) for v in row.split(",") if v])
        input_data = np.array(data)
    elif content_type == "application/json":
        input_data = np.array(json.loads(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

    predictions = _model.predict(input_data)

    output = Output()
    output.add_as_json(predictions.tolist())
    return output
