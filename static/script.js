
// ============================================================
// SENTINEL AI FRAUD INTELLIGENCE
// Frontend Controller
// ============================================================

const API_URL = "http://127.0.0.1:8000";

// ------------------------------------------------------------
// DOM READY
// ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {

    initializeTheme();
    initializeNavigation();
    initializeForm();
    initializeNotifications();

    checkAPIHealth();

});


// ============================================================
// API HEALTH
// ============================================================

async function checkAPIHealth() {

    const apiStatus = document.getElementById("apiStatus");
    const systemStatus = document.getElementById("systemStatus");
    const sidebarStatus = document.getElementById("sidebarStatus");
    const analysisStatus = document.getElementById("analysisStatus");

    try {

        const response = await fetch(`${API_URL}/health`, {
            method: "GET"
        });

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        const data = await response.json();

        if (apiStatus) {
            apiStatus.textContent = "ONLINE";
        }

        if (systemStatus) {
            systemStatus.textContent = "ONLINE";
        }

        if (sidebarStatus) {
            sidebarStatus.textContent = "SYSTEM ONLINE";
        }

        if (analysisStatus) {
            analysisStatus.textContent = "Ready for analysis";
        }

        console.log("SENTINEL API:", data);

    } catch (error) {

        console.error("API health check failed:", error);

        if (apiStatus) {
            apiStatus.textContent = "OFFLINE";
        }

        if (systemStatus) {
            systemStatus.textContent = "OFFLINE";
        }

        if (sidebarStatus) {
            sidebarStatus.textContent = "API OFFLINE";
        }

        if (analysisStatus) {
            analysisStatus.textContent = "API connection unavailable";
        }
    }
}


// ============================================================
// FORM INITIALIZATION
// ============================================================

function initializeForm() {

    const form = document.getElementById("transactionForm");

    if (!form) return;

    form.addEventListener("submit", async (event) => {

        event.preventDefault();

        await analyzeTransaction();

    });
}


// ============================================================
// ANALYZE TRANSACTION
// ============================================================

async function analyzeTransaction() {

    const button = document.getElementById("analyzeButton");
    const buttonText = document.getElementById("buttonText");
    const errorBox = document.getElementById("errorBox");
    const errorText = document.getElementById("errorText");

    if (errorBox) {
        errorBox.style.display = "none";
    }

    // --------------------------------------------------------
    // Loading state
    // --------------------------------------------------------

    if (button) {
        button.disabled = true;
    }

    if (buttonText) {
        buttonText.textContent = "Analyzing...";
    }

    updateAnalysisStatus("Running SENTINEL intelligence...");

    try {

        const payload = buildPayload();

        console.log("SENTINEL REQUEST:", payload);

        const response = await fetch(`${API_URL}/analyze`, {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)

        });

        const data = await response.json();

        console.log("SENTINEL RESPONSE:", data);

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.message ||
                "Transaction analysis failed."
            );

        }

        updateDashboard(data);

        showToast("Transaction analysis completed.");

        updateAnalysisStatus("Analysis completed");

    } catch (error) {

        console.error("SENTINEL ERROR:", error);

        if (errorBox) {
            errorBox.style.display = "block";
        }

        if (errorText) {
            errorText.textContent =
                error.message ||
                "Unable to connect to SENTINEL API.";
        }

        updateAnalysisStatus("Analysis failed");

    } finally {

        if (button) {
            button.disabled = false;
        }

        if (buttonText) {
            buttonText.textContent = "Analyze Transaction";
        }
    }
}


// ============================================================
// BUILD API PAYLOAD
// ============================================================

function buildPayload() {

    return {

        TransactionDT:
            numberValue("TransactionDT", 0),

        TransactionAmt:
            numberValue("TransactionAmt", 0),

        card1:
            numberValue("card1", 0),

        card2:
            numberOrNull("card2"),

        card3:
            numberOrNull("card3"),

        card5:
            numberOrNull("card5"),

        addr1:
            numberOrNull("addr1"),

        addr2:
            numberOrNull("addr2"),

        dist1:
            numberOrNull("dist1"),

        ProductCD:
            stringValue("ProductCD", "W"),

        card4:
            stringValue("card4", "visa"),

        card6:
            stringValue("card6", "credit"),

        P_emaildomain:
            stringOrEmpty("P_emaildomain"),

        R_emaildomain:
            stringOrEmpty("R_emaildomain")

    };
}


// ============================================================
// INPUT HELPERS
// ============================================================

function numberValue(id, fallback = 0) {

    const element = document.getElementById(id);

    if (!element) {
        return fallback;
    }

    const value = parseFloat(element.value);

    return Number.isFinite(value) ? value : fallback;
}


function numberOrNull(id) {

    const element = document.getElementById(id);

    if (!element) {
        return null;
    }

    const value = element.value.trim();

    if (value === "") {
        return null;
    }

    const number = parseFloat(value);

    return Number.isFinite(number) ? number : null;
}


function stringValue(id, fallback = "") {

    const element = document.getElementById(id);

    if (!element) {
        return fallback;
    }

    const value = element.value.trim();

    return value || fallback;
}


function stringOrEmpty(id) {

    const element = document.getElementById(id);

    if (!element) {
        return "";
    }

    return element.value.trim();
}


// ============================================================
// UPDATE EVERYTHING
// ============================================================

function updateDashboard(data) {

    updateHero(data);

    updatePipeline(data);

    updateAssessment(data);

    updateSignals(data);

    updateInsight(data);

    updateModelSignal(data);

    updateStatus(data);

}


// ============================================================
// HERO
// ============================================================

function updateHero(data) {

    const score = safeNumber(data.combined_risk_score);

    const riskLevel =
        String(data.risk_level || "LOW").toUpperCase();

    const decision =
        String(data.decision || "APPROVE").toUpperCase();

    const scoreElement =
        document.getElementById("overallRiskScore");

    const badge =
        document.getElementById("heroRiskBadge");

    const decisionElement =
        document.getElementById("heroDecision");

    const explanation =
        document.getElementById("heroExplanation");

    const gauge =
        document.getElementById("riskGauge");


    if (scoreElement) {
        scoreElement.textContent = score.toFixed(2);
    }


    if (badge) {

        badge.textContent = riskLevel;

        badge.className = "risk-badge";

        badge.classList.add(
            riskLevel.toLowerCase()
        );
    }


    if (decisionElement) {
        decisionElement.textContent = decision;
    }


    if (explanation) {

        explanation.textContent =
            data.explanation ||
            getDecisionExplanation(
                decision,
                riskLevel
            );
    }


    if (gauge) {

        const percentage =
            clamp(score, 0, 100);

        gauge.style.setProperty(
            "--risk-score",
            `${percentage}%`
        );
    }
}


// ============================================================
// PIPELINE
// ============================================================

function updatePipeline(data) {

    const ai =
        safeNumber(data.ai_risk_score);

    const behavioral =
        safeNumber(data.behavioral_risk);

    const combined =
        safeNumber(data.combined_risk_score);

    const decision =
        String(data.decision || "APPROVE").toUpperCase();


    const pipelineAI =
        document.getElementById("pipelineAI");

    const pipelineBehavior =
        document.getElementById("pipelineRule");

    const pipelineCombined =
        document.getElementById("pipelineCombined");

    const pipelineDecision =
        document.getElementById("pipelineDecision");

    const decisionIcon =
        document.getElementById("pipelineDecisionIcon");


    if (pipelineAI) {
        pipelineAI.textContent =
            `${ai.toFixed(2)}%`;
    }


    if (pipelineBehavior) {
        pipelineBehavior.textContent =
            `${behavioral.toFixed(2)}`;
    }


    if (pipelineCombined) {
        pipelineCombined.textContent =
            `${combined.toFixed(2)}`;
    }


    if (pipelineDecision) {

        pipelineDecision.textContent =
            decision;
    }


    if (decisionIcon) {

        decisionIcon.textContent =
            getDecisionIcon(decision);
    }
}


// ============================================================
// RISK ASSESSMENT
// ============================================================

function updateAssessment(data) {

    const ai =
        safeNumber(data.ai_risk_score);

    const behavioral =
        safeNumber(data.behavioral_risk);

    const combined =
        safeNumber(data.combined_risk_score);

    const decision =
        String(data.decision || "APPROVE").toUpperCase();


    const aiProbability =
        document.getElementById("aiProbability");

    const aiProgress =
        document.getElementById("aiProgress");

    const ruleScore =
        document.getElementById("ruleScore");

    const ruleProgress =
        document.getElementById("ruleProgress");

    const combinedScore =
        document.getElementById("combinedScore");

    const combinedProgress =
        document.getElementById("combinedProgress");

    const decisionBox =
        document.getElementById("decisionBox");

    const decisionValue =
        document.getElementById("decisionValue");

    const decisionDescription =
        document.getElementById("decisionDescription");

    const marker =
        document.getElementById("breakdownMarker");


    // AI

    if (aiProbability) {

        aiProbability.textContent =
            `${ai.toFixed(2)}%`;
    }

    if (aiProgress) {

        aiProgress.style.width =
            `${clamp(ai, 0, 100)}%`;
    }


    // Behavioral

    if (ruleScore) {

        ruleScore.textContent =
            behavioral.toFixed(2);
    }

    if (ruleProgress) {

        ruleProgress.style.width =
            `${clamp(behavioral, 0, 100)}%`;
    }


    // Combined

    if (combinedScore) {

        combinedScore.textContent =
            combined.toFixed(2);
    }

    if (combinedProgress) {

        combinedProgress.style.width =
            `${clamp(combined, 0, 100)}%`;
    }


    // Decision

    if (decisionValue) {

        decisionValue.textContent =
            decision;

        decisionValue.classList.remove(
            "approve",
            "review",
            "block",
            "low",
            "medium",
            "high"
        );

        decisionValue.classList.add(
            decision.toLowerCase()
        );
    }


    if (decisionDescription) {

        decisionDescription.textContent =
            getDecisionExplanation(
                decision,
                data.risk_level
            );
    }


    if (decisionBox) {

        decisionBox.classList.remove(
            "approve",
            "review",
            "block"
        );

        decisionBox.classList.add(
            decision.toLowerCase()
        );
    }


    if (marker) {

        marker.style.left =
            `${clamp(combined, 0, 100)}%`;
    }
}


// ============================================================
// RISK SIGNALS
// ============================================================

function updateSignals(data) {

    const container =
        document.getElementById("signalsList");

    const count =
        document.getElementById("signalCount");


    if (!container) {
        return;
    }


    const signals =
        Array.isArray(data.risk_signals)
            ? [...data.risk_signals]
            : [];


    if (count) {
        count.textContent = signals.length;
    }


    container.innerHTML = "";


    if (signals.length === 0) {

        container.innerHTML = `
            <div class="signal-item low">
                <div class="signal-icon">✓</div>
                <div class="signal-content">
                    <div class="signal-title">
                        No elevated risk signals
                    </div>
                    <div class="signal-description">
                        SENTINEL did not identify a material
                        intervention-grade risk signal.
                    </div>
                </div>
                <div class="signal-severity low">
                    LOW
                </div>
            </div>
        `;

        return;
    }


    signals.forEach((signal) => {

        const severity =
            classifySignalSeverity(
                signal,
                data
            );

        const icon =
            getSignalIcon(severity);


        const item =
            document.createElement("div");

        item.className =
            `signal-item ${severity.toLowerCase()}`;


        item.innerHTML = `

            <div class="signal-icon">
                ${icon}
            </div>

            <div class="signal-content">

                <div class="signal-title">
                    ${escapeHTML(
                        formatSignalTitle(signal)
                    )}
                </div>

                <div class="signal-description">
                    ${escapeHTML(
                        signalDescription(
                            signal,
                            severity
                        )
                    )}
                </div>

            </div>

            <div class="signal-severity ${severity.toLowerCase()}">
                ${severity}
            </div>
        `;


        container.appendChild(item);

    });
}


// ============================================================
// SIGNAL SEVERITY
// ============================================================

function classifySignalSeverity(signal, data) {

    const text =
        String(signal || "").toLowerCase();

    const combined =
        safeNumber(data.combined_risk_score);

    const ai =
        safeNumber(data.ai_risk_score);

    const decision =
        String(data.decision || "APPROVE").toUpperCase();


    // --------------------------------------------------------
    // CRITICAL / HIGH
    // --------------------------------------------------------

    if (
        decision === "BLOCK" &&
        (
            text.includes("fraud probability") ||
            text.includes("fraud") ||
            text.includes("suspicious")
        )
    ) {
        return "HIGH";
    }


    // --------------------------------------------------------
    // MEDIUM
    // --------------------------------------------------------

    if (decision === "REVIEW") {

        if (
            text.includes("fraud") ||
            text.includes("unusual") ||
            text.includes("elevated") ||
            text.includes("distance") ||
            text.includes("linked") ||
            text.includes("association")
        ) {
            return "MEDIUM";
        }
    }


    if (decision === "BLOCK") {

        if (
            ai >= 45 ||
            combined >= 60
        ) {
            return "HIGH";
        }

        return "MEDIUM";
    }


    // --------------------------------------------------------
    // APPROVED / LOW TRANSACTIONS
    //
    // Supporting signals should NOT visually look like
    // intervention-grade alerts.
    // --------------------------------------------------------

    if (
        decision === "APPROVE" &&
        combined < 30
    ) {
        return "LOW";
    }


    if (combined < 50) {
        return "LOW";
    }


    return "MEDIUM";
}


// ============================================================
// SIGNAL TITLE
// ============================================================

function formatSignalTitle(signal) {

    let text =
        String(signal || "").trim();


    if (!text) {
        return "Risk signal";
    }


    text =
        text.replace(
            /^SENTINEL\s*/i,
            ""
        );


    return text;
}


// ============================================================
// SIGNAL DESCRIPTION
// ============================================================

function signalDescription(signal, severity) {

    const text =
        String(signal || "").toLowerCase();


    if (severity === "LOW") {

        if (text.includes("amount")) {
            return "Amount pattern detected as supporting context; current risk remains below intervention level.";
        }

        if (text.includes("time")) {
            return "Transaction timing differs from the learned pattern but is not sufficient for intervention.";
        }

        if (text.includes("distance")) {
            return "Distance information contributes contextual evidence but does not independently indicate fraud.";
        }

        if (
            text.includes("historical") ||
            text.includes("association") ||
            text.includes("linked")
        ) {
            return "Historical relationship evidence is present as supporting intelligence, while the final risk remains low.";
        }

        return "Supporting intelligence detected; it does not independently justify intervention.";
    }


    if (severity === "MEDIUM") {

        return "This signal contributes meaningful risk evidence and is considered alongside the champion AI model.";
    }


    return "This signal contributes significant evidence to the transaction risk assessment.";
}


// ============================================================
// INSIGHT
// ============================================================

function updateInsight(data) {

    const title =
        document.getElementById("insightTitle");

    const text =
        document.getElementById("insightText");


    if (!title || !text) {
        return;
    }


    const decision =
        String(data.decision || "APPROVE").toUpperCase();

    const risk =
        String(data.risk_level || "LOW").toUpperCase();

    const ai =
        safeNumber(data.ai_fraud_probability);


    if (decision === "BLOCK") {

        title.textContent =
            "Intervention recommended";

        text.textContent =
            `The champion AI model estimated ${(ai * 100).toFixed(2)}% fraud probability, exceeding the configured operating threshold. Supporting behavioral and relationship intelligence strengthens the decision.`;

        return;
    }


    if (decision === "REVIEW") {

        title.textContent =
            "Manual review recommended";

        text.textContent =
            `The transaction shows meaningful risk evidence, but the combined assessment does not provide sufficient confidence for automatic blocking.`;

        return;
    }


    title.textContent =
        "Transaction appears low risk";

    text.textContent =
        `The champion AI model estimated ${(ai * 100).toFixed(2)}% fraud probability. Supporting intelligence was considered, but the transaction remains below the intervention threshold.`;
}


// ============================================================
// MODEL SIGNAL
// ============================================================

function updateModelSignal(data) {

    const modelSignal =
        document.getElementById("modelSignal");

    const track =
        document.getElementById("modelSignalTrack");


    if (!modelSignal) {
        return;
    }


    const ai =
        safeNumber(data.ai_fraud_probability);

    const threshold =
        safeNumber(
            data.threshold,
            0.45
        );


    modelSignal.textContent =
        `Champion model probability ${(ai * 100).toFixed(2)}% · operating threshold ${(threshold * 100).toFixed(0)}%`;


    if (track) {

        track.style.width =
            `${clamp(ai * 100, 0, 100)}%`;
    }
}


// ============================================================
// STATUS
// ============================================================

function updateStatus(data) {

    const model =
        document.getElementById("modelSignal");

    const system =
        document.getElementById("systemStatus");

    const protection =
        document.getElementById("protectionText");


    if (system) {
        system.textContent = "ONLINE";
    }


    if (protection) {

        const decision =
            String(
                data.decision || "APPROVE"
            ).toUpperCase();

        protection.textContent =
            decision === "BLOCK"
                ? "Threat blocked"
                : decision === "REVIEW"
                    ? "Review required"
                    : "Transaction protected";
    }
}


// ============================================================
// DECISION EXPLANATION
// ============================================================

function getDecisionExplanation(
    decision,
    riskLevel
) {

    decision =
        String(decision || "APPROVE").toUpperCase();


    if (decision === "BLOCK") {

        return "SENTINEL blocked the transaction because the champion AI fraud model exceeded its validated operating threshold.";
    }


    if (decision === "REVIEW") {

        return "SENTINEL recommends manual review because the transaction contains meaningful risk evidence without sufficient confidence for automatic blocking.";
    }


    return "Transaction may proceed. The primary AI fraud probability remains below the intervention threshold.";
}


// ============================================================
// ICONS
// ============================================================

function getDecisionIcon(decision) {

    switch (decision) {

        case "BLOCK":
            return "✕";

        case "REVIEW":
            return "!";

        default:
            return "✓";
    }
}


function getSignalIcon(severity) {

    switch (severity) {

        case "HIGH":
            return "⚠";

        case "MEDIUM":
            return "•";

        default:
            return "✓";
    }
}


// ============================================================
// THEME
// ============================================================

function initializeTheme() {

    const button =
        document.getElementById("themeButton");

    if (!button) {
        return;
    }


    button.addEventListener("click", () => {

        document.body.classList.toggle("light-theme");

        const light =
            document.body.classList.contains(
                "light-theme"
            );

        localStorage.setItem(
            "sentinel-theme",
            light ? "light" : "dark"
        );
    });


    const saved =
        localStorage.getItem(
            "sentinel-theme"
        );


    if (saved === "light") {

        document.body.classList.add(
            "light-theme"
        );
    }
}


// ============================================================
// NAVIGATION
// ============================================================

function initializeNavigation() {

    const navItems =
        document.querySelectorAll(
            ".nav-item"
        );


    navItems.forEach((item) => {

        item.addEventListener(
            "click",
            (event) => {

                const href =
                    item.getAttribute("href");


                if (
                    href &&
                    href.startsWith("#")
                ) {

                    const target =
                        document.querySelector(
                            href
                        );


                    if (target) {

                        event.preventDefault();

                        target.scrollIntoView({
                            behavior: "smooth",
                            block: "start"
                        });
                    }
                }


                navItems.forEach(
                    (nav) =>
                        nav.classList.remove(
                            "active"
                        )
                );


                item.classList.add(
                    "active"
                );
            }
        );

    });
}


// ============================================================
// NOTIFICATIONS
// ============================================================

function initializeNotifications() {

    const button =
        document.getElementById(
            "notificationButton"
        );


    if (!button) {
        return;
    }


    button.addEventListener(
        "click",
        () => {

            showToast(
                "SENTINEL monitoring is active."
            );
        }
    );
}


// ============================================================
// TOAST
// ============================================================

function showToast(message) {

    const toast =
        document.getElementById("toast");

    const toastText =
        document.getElementById("toastText");


    if (!toast) {
        return;
    }


    if (toastText) {
        toastText.textContent = message;
    }


    toast.classList.add("show");


    setTimeout(() => {

        toast.classList.remove("show");

    }, 3000);
}


// ============================================================
// ANALYSIS STATUS
// ============================================================

function updateAnalysisStatus(message) {

    const element =
        document.getElementById(
            "analysisStatus"
        );


    if (element) {
        element.textContent = message;
    }
}


// ============================================================
// SAFE NUMBER
// ============================================================

function safeNumber(value, fallback = 0) {

    const number =
        Number(value);

    return Number.isFinite(number)
        ? number
        : fallback;
}


// ============================================================
// CLAMP
// ============================================================

function clamp(value, min, max) {

    return Math.min(
        Math.max(
            safeNumber(value),
            min
        ),
        max
    );
}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHTML(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
