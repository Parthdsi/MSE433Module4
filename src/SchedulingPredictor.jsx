import React, { useState } from "react";

const API_BASE = "http://localhost:5001";

const PHYSICIANS = ["Dr. A", "Dr. B", "Dr. C"];
const CASE_POSITIONS = [
  { value: 1, label: "1st" },
  { value: 2, label: "2nd" },
  { value: 3, label: "3rd" },
  { value: 4, label: "4th" },
  { value: 5, label: "5th+" },
];
const COMPLEXITY_OPTIONS = [
  { value: "Standard PVI", label: "Standard PVI" },
  { value: "BOX or PST BOX", label: "BOX / PST BOX" },
  { value: "CTI or SVC", label: "CTI / SVC" },
  { value: "AAFL", label: "AAFL" },
];

function Toggle({ label, value, onChange, defaultVal }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-100">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <button
        type="button"
        onClick={() => onChange(value === "Y" ? "N" : "Y")}
        className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 ${
          value === "Y"
            ? "bg-blue-600 focus:ring-blue-500"
            : "bg-gray-300 focus:ring-gray-400"
        }`}
      >
        <span
          className={`inline-block h-6 w-6 transform rounded-full bg-white shadow transition-transform ${
            value === "Y" ? "translate-x-7" : "translate-x-1"
          }`}
        />
        <span className="sr-only">{label}</span>
      </button>
    </div>
  );
}

function ResultCard({ result }) {
  if (!result) return null;
  const mins = result.predicted_minutes;
  const lo = Math.round(mins - 15);
  const hi = Math.round(mins + 15);

  let color, bg, border, verdict;
  if (mins < 80) {
    color = "text-green-800";
    bg = "bg-green-50";
    border = "border-green-200";
    verdict = "Standard case — schedule normally";
  } else if (mins <= 110) {
    color = "text-yellow-800";
    bg = "bg-yellow-50";
    border = "border-yellow-200";
    verdict = "Allow extra buffer — consider position in day";
  } else {
    color = "text-red-800";
    bg = "bg-red-50";
    border = "border-red-200";
    verdict = "Complex case — schedule first or allow 2-hour block";
  }

  return (
    <div className={`mt-6 rounded-xl border-2 ${border} ${bg} p-6`}>
      <div className="text-center">
        <p className="text-sm font-medium text-gray-500 uppercase tracking-wide">
          Predicted Duration
        </p>
        <p className={`text-5xl font-bold mt-1 ${color}`}>
          {Math.round(mins)} <span className="text-2xl font-normal">min</span>
        </p>
        <p className="text-gray-500 mt-1 text-sm">
          Confidence range: {lo}–{hi} min (±15 min)
        </p>
      </div>
      <div className={`mt-4 rounded-lg p-3 ${bg} border ${border}`}>
        <p className={`text-center font-semibold ${color}`}>{verdict}</p>
      </div>
    </div>
  );
}

export default function SchedulingPredictor() {
  const [mode, setMode] = useState("A");
  const [physician, setPhysician] = useState("Dr. A");
  const [casePosition, setCasePosition] = useState(1);
  const [complexity, setComplexity] = useState("Standard PVI");

  const [obesity, setObesity] = useState("N");
  const [sleepApnea, setSleepApnea] = useState("N");
  const [fastingNotConfirmed, setFastingNotConfirmed] = useState("N");
  const [bloodworkIncomplete, setBloodworkIncomplete] = useState("N");
  const [equipmentPrestaged, setEquipmentPrestaged] = useState("Y");
  const [anesthesiaReady, setAnesthesiaReady] = useState("Y");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const firstCase = casePosition === 1 ? "Y" : "N";

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const endpoint =
      mode === "A" ? "/predict/model1" : "/predict/model2";

    const body =
      mode === "A"
        ? { physician, case_of_day: casePosition, complexity }
        : {
            physician,
            case_of_day: casePosition,
            complexity,
            first_case_of_day: firstCase,
            obesity,
            sleep_apnea: sleepApnea,
            fasting_not_confirmed: fastingNotConfirmed,
            bloodwork_incomplete: bloodworkIncomplete,
            equipment_prestaged: equipmentPrestaged,
            anesthesia_ready: anesthesiaReady,
          };

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const isModeB = mode === "B";

  return (
    <div className={`min-h-screen ${isModeB ? "bg-amber-50" : "bg-slate-50"} transition-colors duration-300`}>
      <div className="max-w-lg mx-auto px-4 py-6">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900">
            EP Lab Scheduling
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            AFib Ablation Duration Predictor
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-1 flex mb-6">
          <button
            type="button"
            onClick={() => { setMode("A"); setResult(null); }}
            className={`flex-1 py-3 px-4 rounded-lg text-sm font-semibold transition-all ${
              mode === "A"
                ? "bg-blue-600 text-white shadow-md"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Today (Current Model)
          </button>
          <button
            type="button"
            onClick={() => { setMode("B"); setResult(null); }}
            className={`flex-1 py-3 px-4 rounded-lg text-sm font-semibold transition-all ${
              mode === "B"
                ? "bg-amber-500 text-white shadow-md"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            Future State (App-Enhanced)
          </button>
        </div>

        {/* Future State Banner */}
        {isModeB && (
          <div className="bg-yellow-100 border border-yellow-300 rounded-lg p-3 mb-6">
            <p className="text-yellow-800 text-sm text-center font-medium">
              Future state model — based on simulated data. Deploy the logging
              app to validate.
            </p>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5 space-y-5">
            {/* Physician */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Physician
              </label>
              <div className="grid grid-cols-3 gap-2">
                {PHYSICIANS.map((doc) => (
                  <button
                    key={doc}
                    type="button"
                    onClick={() => setPhysician(doc)}
                    className={`py-3 rounded-lg text-sm font-semibold border-2 transition-all ${
                      physician === doc
                        ? "border-blue-500 bg-blue-50 text-blue-700"
                        : "border-gray-200 bg-gray-50 text-gray-600 hover:border-gray-300"
                    }`}
                  >
                    {doc}
                  </button>
                ))}
              </div>
            </div>

            {/* Case Position */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Case Position in Day
              </label>
              <div className="grid grid-cols-5 gap-2">
                {CASE_POSITIONS.map((pos) => (
                  <button
                    key={pos.value}
                    type="button"
                    onClick={() => setCasePosition(pos.value)}
                    className={`py-3 rounded-lg text-sm font-semibold border-2 transition-all ${
                      casePosition === pos.value
                        ? "border-blue-500 bg-blue-50 text-blue-700"
                        : "border-gray-200 bg-gray-50 text-gray-600 hover:border-gray-300"
                    }`}
                  >
                    {pos.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Complexity */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Procedure Complexity
              </label>
              <div className="grid grid-cols-2 gap-2">
                {COMPLEXITY_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    onClick={() => setComplexity(opt.value)}
                    className={`py-3 px-3 rounded-lg text-sm font-semibold border-2 transition-all ${
                      complexity === opt.value
                        ? "border-blue-500 bg-blue-50 text-blue-700"
                        : "border-gray-200 bg-gray-50 text-gray-600 hover:border-gray-300"
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Mode B extra fields */}
            {isModeB && (
              <div className="border-t border-gray-200 pt-4 mt-4">
                <p className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-3">
                  App-Logged Pre-Case Flags
                </p>
                <Toggle label="Obesity" value={obesity} onChange={setObesity} />
                <Toggle
                  label="Sleep Apnea"
                  value={sleepApnea}
                  onChange={setSleepApnea}
                />
                <Toggle
                  label="Fasting Not Confirmed"
                  value={fastingNotConfirmed}
                  onChange={setFastingNotConfirmed}
                />
                <Toggle
                  label="Bloodwork Incomplete"
                  value={bloodworkIncomplete}
                  onChange={setBloodworkIncomplete}
                />
                <Toggle
                  label="Equipment Pre-staged"
                  value={equipmentPrestaged}
                  onChange={setEquipmentPrestaged}
                />
                <Toggle
                  label="Anesthesia Team Ready"
                  value={anesthesiaReady}
                  onChange={setAnesthesiaReady}
                />
              </div>
            )}
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className={`w-full mt-6 py-4 rounded-xl text-white font-bold text-lg shadow-lg transition-all active:scale-[0.98] ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : isModeB
                ? "bg-amber-500 hover:bg-amber-600"
                : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? "Predicting..." : "Predict Duration"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-xl">
            <p className="text-red-700 text-sm text-center">{error}</p>
          </div>
        )}

        {/* Result */}
        <ResultCard result={result} />

        {/* Footer */}
        <p className="text-center text-xs text-gray-400 mt-8">
          MSE 433 — EP Lab Scheduling Predictor
        </p>
      </div>
    </div>
  );
}
