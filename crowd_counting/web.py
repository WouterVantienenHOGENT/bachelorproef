"""Local Streamlit interface with a reusable inference model per session."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import time
from datetime import datetime

from PIL import Image
import streamlit as st

from model_catalog import MODEL_DIR, ModelEntry, discover_models


ROOT = Path(__file__).resolve().parents[1]
WORKDIR = ROOT / "bachproef"


def local_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKDIR / path


def model_key(checkpoint: Path, device: str, architecture: str = "auto"):
    path = checkpoint.resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size, device, architecture


def get_model(checkpoint: Path, device: str, architecture: str = "auto"):
    key = model_key(checkpoint, device, architecture)
    if st.session_state.get("loaded_model_key") == key:
        return st.session_state.loaded_model, True
    # Retain only the selected model in this browser session.
    st.session_state.pop("loaded_model", None)
    st.session_state.pop("loaded_model_key", None)
    from infer import load_model

    loaded = load_model(checkpoint, device, architecture)
    st.session_state.loaded_model = loaded
    st.session_state.loaded_model_key = key
    return loaded, False


def run_inference(checkpoint, image_bytes, image_name, output_root, device,
                  warmup_runs, timed_runs, benchmark_csv, architecture="auto",
                  max_edge=1024, threshold=0.5, model_name=""):
    """Save the input and run shared inference with a persistent model."""
    started = time.perf_counter()
    if not checkpoint.is_file():
        raise ValueError("Het gekozen checkpoint bestaat niet.")
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(
        prefix=datetime.now().strftime("%Y%m%d-%H%M%S-"), dir=output_root,
    ))
    inputs = output / "input"
    inputs.mkdir()
    image_path = inputs / Path(image_name).name
    image_path.write_bytes(image_bytes)
    try:
        load_started = time.perf_counter()
        loaded, reused = get_model(checkpoint, device, architecture)
        load_ms = 1000 * (time.perf_counter() - load_started)
        from infer import InferenceOptions, predict_image

        result = predict_image(InferenceOptions(
            checkpoint=str(checkpoint), image=str(image_path), output_dir=str(output),
            device=device, warmup_runs=warmup_runs, timed_runs=timed_runs,
            benchmark_csv=str(benchmark_csv) if benchmark_csv else "",
            architecture=architecture, max_edge=max_edge, threshold=threshold,
        ), loaded)
        result.update(model_reused=reused, model_load_ms=load_ms, model_name=model_name or checkpoint.stem)
        log = (
            f"Geschat aantal personen: {result['estimated_count']:.2f}\n"
            f"Modeltijd: {result['mean_forward_ms']:.2f} ms\n"
            f"Model hergebruikt: {reused}\n"
        )
        (output / "inference.log").write_text(log, encoding="utf-8")
        result["web_processing_ms"] = 1000 * (time.perf_counter() - started)
        (output / f"{image_path.stem}_inference.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    except Exception as exc:
        (output / "inference.log").write_text(str(exc), encoding="utf-8")
        raise RuntimeError(f"Inferentie mislukt: {exc}\nLogbestand: {output / 'inference.log'}") from exc
    stem = image_path.stem
    files = {name: (output / name).read_bytes() for name in result["exports"]}
    return {
        "result": json.loads(files[f"{stem}_inference.json"]),
        "files": files, "input": image_bytes, "name": image_path.name,
        "output": str(output), "log": log,
    }


def main():
    st.set_page_config(page_title="Personen tellen", layout="wide")
    st.title("Personen tellen")
    st.write("Kies een model en een afbeelding om het aantal personen te schatten.")
    image_tab, live_tab = st.tabs(["Afbeelding", "Livestream · later"])
    with live_tab:
        st.info("Livebeelden analyseren wordt later toegevoegd.")
        st.text_input("Link naar livestream", placeholder="https://…", disabled=True)
        st.button("Livestream starten", disabled=True)

    with image_tab:
        controls, preview = st.columns([1, 1])
        with controls:
            with st.expander("Modellen beheren"):
                models_dir = st.text_input("Modellenmap", value=str(MODEL_DIR))
                st.caption("Kopieer .pt- of .pth-gewichten naar csrnet/ of p2pnet/ in deze map. "
                           "Gebruik een herkenbare bestandsnaam, of een gelijknamig JSON-bestand met een naam.")
                st.button("Modellen vernieuwen")
            try:
                entries, errors = discover_models(local_path(models_dir)) if models_dir.strip() else ([], [])
            except OSError as exc:
                entries, errors = [], [str(exc)]
            for error in errors:
                st.warning(error)
            chosen = st.selectbox(
                "Model", [*entries, "Ander checkpoint…"],
                index=next((i for i, entry in enumerate(entries) if entry.path.stem == "csrnet_finaal"), 0),
                format_func=lambda entry: (entry.name if sum(e.name == entry.name for e in entries) == 1
                                           else f"{entry.name} ({entry.path.relative_to(local_path(models_dir))})")
                if isinstance(entry, ModelEntry) else entry,
            )
            if isinstance(chosen, ModelEntry):
                checkpoint, architecture, model_name = chosen.path, chosen.architecture, chosen.name
                if chosen.description:
                    st.caption(chosen.description)
            else:
                custom = st.text_input("Pad naar checkpoint", placeholder="../models/p2pnet/mijn_model.pth")
                checkpoint = local_path(custom) if custom.strip() else None
                architecture = st.selectbox("Modeltype", ["auto", "csrnet", "p2pnet"])
                model_name = checkpoint.stem if checkpoint else ""
            if not entries:
                st.info("Nog geen modellen gevonden. Voer crowd_counting/setup_models.py uit of kopieer gewichten naar de modellenmap.")
            device = st.selectbox("Rekenapparaat", ["auto", "cpu", "cuda"],
                                   help="Auto kiest CUDA als die beschikbaar is, anders CPU.")
            if st.button("Model vooraf laden", disabled=checkpoint is None):
                try:
                    started = time.perf_counter()
                    with st.spinner("Model laden…"):
                        _, reused = get_model(checkpoint, device, architecture)
                    st.success("Model was al geladen." if reused else
                               f"Model geladen in {time.perf_counter() - started:.2f} s.")
                except Exception as exc:
                    st.error(f"Kan het model niet laden: {exc}")
            model_status = st.empty()
            st.caption("Het model blijft geladen voor volgende voorspellingen in deze sessie. Vooraf laden is optioneel.")
            source = st.radio("Afbeeldingsbron", ["Uploaden", "Lokaal bestand"], horizontal=True)
            image_bytes = None
            image_name = ""
            try:
                if source == "Uploaden":
                    upload = st.file_uploader("Afbeelding uploaden", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"])
                    if upload is not None:
                        image_bytes, image_name = upload.getvalue(), upload.name
                else:
                    image_value = st.text_input("Pad naar afbeelding")
                    if image_value.strip():
                        image_path = local_path(image_value)
                        image_bytes, image_name = image_path.read_bytes(), image_path.name
                if image_bytes is not None:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        with preview:
                            st.image(image.convert("RGB"), caption=image_name, width="stretch")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                image_bytes = None
                st.error(f"Kan de afbeelding niet openen: {exc}")

            with st.expander("Instellingen en opslag"):
                max_edge = st.selectbox("Maximale beeldzijde voor inferentie", [1024, 1536, 2048, 0],
                                       format_func=lambda value: f"{value} pixels" if value else "Oorspronkelijke resolutie",
                                       help="Meer detail vraagt meer geheugen en rekentijd. P2PNet rondt beide afmetingen af op veelvouden van 128.")
                threshold = st.slider("P2PNet-betrouwbaarheidsgrens", 0.0, 1.0, 0.5, 0.05,
                                      disabled=architecture == "csrnet",
                                      help="De officiële P2PNet-demo telt punten met een score boven 0,5.")
                warmup_runs = st.number_input("Opwarmruns", min_value=0, value=0, step=1)
                timed_runs = st.number_input("Gemeten runs", min_value=1, value=1, step=1)
                output_value = st.text_input("Outputmap", value="predictions/web")
                save_benchmark = st.checkbox("Benchmark toevoegen aan CSV", value=False)
                csv_value = st.text_input("Benchmark-CSV", value="runs/web_inference_benchmarks.csv", disabled=not save_benchmark)
                st.caption("Relatieve paden vertrekken vanuit de map bachproef. Iedere uitvoering krijgt een eigen submap met de invoer en resultaten.")
            if st.button("Voorspelling maken", type="primary", disabled=image_bytes is None or checkpoint is None):
                st.session_state.pop("prediction", None)
                try:
                    if not output_value.strip():
                        raise ValueError("Vul een outputmap in.")
                    if save_benchmark and not csv_value.strip():
                        raise ValueError("Vul een CSV-pad in of schakel de benchmarkopslag uit.")
                    with st.spinner("Model laden en afbeelding verwerken…"):
                        st.session_state.prediction = run_inference(
                            checkpoint, image_bytes, image_name, local_path(output_value),
                            device, warmup_runs, timed_runs,
                            local_path(csv_value) if save_benchmark else None,
                            architecture, max_edge, threshold, model_name,
                        )
                except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
                    st.error(str(exc))

            try:
                ready = checkpoint is not None and st.session_state.get("loaded_model_key") == model_key(checkpoint, device, architecture)
            except OSError:
                ready = False
            model_status.caption("Gekozen model is geladen." if ready else "Gekozen model is nog niet geladen.")

        prediction = st.session_state.get("prediction")
        if prediction:
            st.divider()
            st.subheader("Laatste voltooide voorspelling")
            result = prediction["result"]
            st.caption(f"{prediction['name']} · {result.get('model_name', result['checkpoint'])} · {result['timestamp_utc']}")
            st.caption(f"Invoerresolutie: {result['image_size'][0]} × {result['image_size'][1]} pixels · "
                       f"Origineel: {result['original_image_size'][0]} × {result['original_image_size'][1]} pixels")
            count, latency, total = st.columns(3)
            count.metric("Geschat aantal personen", str(int(result["estimated_count"]))
                         if result.get("prediction_type") == "points" else f"{result['estimated_count']:.2f}")
            latency.metric("Gemiddelde modeltijd", f"{result['mean_forward_ms']:.2f} ms")
            total.metric("Totale verwerking", f"{result.get('web_processing_ms', result['total_command_ms']) / 1000:.2f} s")
            st.caption(
                f"Model laden: {result.get('model_load_ms', 0) / 1000:.2f} s · "
                + ("model hergebruikt. " if result.get('model_reused', False) else "model nieuw geladen. ")
                + "Modeltijd meet alleen de berekening. Totale verwerking omvat ook laden, "
                "beeldvoorbereiding en opslag, maar niet uploaden of weergave in de browser."
            )
            original, density = st.columns(2)
            original.image(prediction["input"], caption="Geanalyseerde afbeelding", width="stretch")
            png = next(data for name, data in prediction["files"].items() if name.endswith(".png"))
            density.image(png, caption=result.get("visualization_caption", "Voorspelde density map"), width="stretch")
            if result.get("points_outside_image", 0):
                st.caption(f"{result['points_outside_image']} getelde punten vallen buiten de foto. "
                           "Ze blijven meetellen volgens de officiële P2PNet-telregel en staan in de CSV.")
            st.caption(f"Opgeslagen in {prediction['output']}")
            for column, (name, data) in zip(st.columns(3), prediction["files"].items()):
                mime = "image/png" if name.endswith(".png") else "application/json" if name.endswith(".json") else "text/csv" if name.endswith(".csv") else "application/octet-stream"
                column.download_button(f"Download {name.rsplit('.', 1)[1].upper()}", data, file_name=name, mime=mime)
            with st.expander("Alle meetgegevens en CLI-uitvoer"):
                st.json(result)
                st.code(prediction["log"], language="text")


if __name__ == "__main__":
    main()
