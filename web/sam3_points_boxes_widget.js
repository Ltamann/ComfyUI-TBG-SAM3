import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

/**
 * Unified SAM3 Prompt Collector Widget
 * - Points: Left-click = positive (green), Right-click/Shift+Left = negative (red)
 * - Boxes: Left-drag = positive (cyan), Right-drag/Shift = negative (red)
 * - Keyboard:
 *     B = toggle Point/Box
 *     Esc = reset to Point(+)
 */

app.registerExtension({
    name: "Comfy.SAM3.PromptCollector",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "TBGSAM3PromptCollector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;

            // Find hidden widget inputs
            const positivePointsWidget = node.widgets.find(w => w.name === "positive_points");
            const negativePointsWidget = node.widgets.find(w => w.name === "negative_points");
            const positiveBoxesWidget = node.widgets.find(w => w.name === "positive_boxes");
            const negativeBoxesWidget = node.widgets.find(w => w.name === "negative_boxes");

            if (positivePointsWidget) positivePointsWidget.type = "hidden";
            if (negativePointsWidget) negativePointsWidget.type = "hidden";
            if (positiveBoxesWidget) positiveBoxesWidget.type = "hidden";
            if (negativeBoxesWidget) negativeBoxesWidget.type = "hidden";

            // Create canvas
            const canvas = document.createElement("canvas");
            canvas.width = 512;
            canvas.height = 512;
            canvas.style.border = "1px solid var(--border-color)";
            canvas.style.cursor = "crosshair";
            canvas.style.backgroundColor = "#1a1a1a";
            canvas.style.display = "block";
            canvas.style.width = "100%";
            canvas.style.height = "auto";
            const ctx = canvas.getContext("2d");

            // Outer wrapper: header row (info+buttons) + canvas
            const canvasWrapper = document.createElement("div");
            canvasWrapper.style.position = "relative";
            canvasWrapper.style.display = "block";
            canvasWrapper.style.width = "100%";
            canvasWrapper.style.height = "auto";
            canvasWrapper.style.overflow = "visible";
            canvasWrapper.style.boxSizing = "border-box";
            canvasWrapper.style.backgroundColor = "#1a1a1a";
            canvasWrapper.style.borderRadius = "6px";
            canvasWrapper.style.padding = "4px";

            const headerRow = document.createElement("div");
            headerRow.style.display = "flex";
            headerRow.style.flexDirection = "row";
            headerRow.style.alignItems = "center";
            headerRow.style.justifyContent = "space-between";
            headerRow.style.marginBottom = "4px";
            canvasWrapper.appendChild(headerRow);

            const modeIndicator = document.createElement("div");
            modeIndicator.style.cssText = `
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 12px;
            `;
            headerRow.appendChild(modeIndicator);

            const buttonsRow = document.createElement("div");
            buttonsRow.style.display = "flex";
            buttonsRow.style.flexDirection = "row";
            buttonsRow.style.gap = "4px";
            headerRow.appendChild(buttonsRow);

            const clearButton = document.createElement("button");
            clearButton.textContent = "Clear";
            clearButton.style.cssText = `
                padding: 2px 6px;
                font-size: 11px;
                border-radius: 4px;
                border: 1px solid #a22;
                background: #d44;
                color: #fff;
                cursor: pointer;
            `;

            const undoButton = document.createElement("button");
            undoButton.textContent = "Undo";
            undoButton.style.cssText = `
                padding: 2px 6px;
                font-size: 11px;
                border-radius: 4px;
                border: 1px solid #222;
                background: #444;
                color: #fff;
                cursor: pointer;
            `;

            buttonsRow.appendChild(clearButton);
            buttonsRow.appendChild(undoButton);

            // Canvas below header
            canvasWrapper.appendChild(canvas);

            const domWidget = node.addDOMWidget("canvas", "canvas", canvasWrapper);
            domWidget.serialize = false;

            // State
            let backgroundImage = null;
            let positivePoints = [];
            let negativePoints = [];
            let positiveBoxes = [];
            let negativeBoxes = [];
            let isDrawingBox = false;
            let boxStartX = 0;
            let boxStartY = 0;
            let currentMode = "point"; // "point" | "box"
            let currentPolarity = "positive"; // "positive" | "negative"
            let isMouseOverCanvas = false;

            // Load saved prompts from hidden widgets (when workflow is loaded)
            const loadFromWidgets = () => {
                const safeParse = (val) => {
                    if (!val || typeof val !== "string") return [];
                    try {
                        const parsed = JSON.parse(val);
                        return Array.isArray(parsed) ? parsed : [];
                    } catch (e) {
                        console.warn("[SAM3 PromptCollector] Failed to parse widget JSON:", e);
                        return [];
                    }
                };

                if (positivePointsWidget && typeof positivePointsWidget.value === "string") {
                    positivePoints = safeParse(positivePointsWidget.value);
                }
                if (negativePointsWidget && typeof negativePointsWidget.value === "string") {
                    negativePoints = safeParse(negativePointsWidget.value);
                }
                if (positiveBoxesWidget && typeof positiveBoxesWidget.value === "string") {
                    positiveBoxes = safeParse(positiveBoxesWidget.value);
                }
                if (negativeBoxesWidget && typeof negativeBoxesWidget.value === "string") {
                    negativeBoxes = safeParse(negativeBoxesWidget.value);
                }
            };

            function updateModeIndicator() {
                const color = currentPolarity === "positive" ? "#00ff00" : "#ff0000";
                modeIndicator.innerHTML =
                    `Mode: <span style="color:${color}">${currentMode.toUpperCase()} (${currentPolarity})</span>` +
                    "<br>Left = Positive, Right/Shift = Negative<br>" +
                    "B = toggle Point/Box, Esc = Point(+)";
            }
            updateModeIndicator();

            function toCanvasCoords(e) {
                const rect = canvas.getBoundingClientRect();
                const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
                const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
                return { x, y };
            }

            function redraw() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                if (backgroundImage) {
                    ctx.drawImage(backgroundImage, 0, 0, canvas.width, canvas.height);
                }

                ctx.lineWidth = 2;
                // positive boxes
                ctx.strokeStyle = "#00ffff";
                positiveBoxes.forEach(box => {
                    const w = box.x2 - box.x1;
                    const h = box.y2 - box.y1;
                    ctx.strokeRect(box.x1, box.y1, w, h);
                });
                // negative boxes
                ctx.strokeStyle = "#ff0000";
                negativeBoxes.forEach(box => {
                    const w = box.x2 - box.x1;
                    const h = box.y2 - box.y1;
                    ctx.strokeRect(box.x1, box.y1, w, h);
                });

                // positive points
                ctx.fillStyle = "#00ff00";
                positivePoints.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });

                // negative points
                ctx.fillStyle = "#ff0000";
                negativePoints.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                });
            }

            function updateWidgets() {
                if (positivePointsWidget) positivePointsWidget.value = JSON.stringify(positivePoints);
                if (negativePointsWidget) negativePointsWidget.value = JSON.stringify(negativePoints);
                if (positiveBoxesWidget) positiveBoxesWidget.value = JSON.stringify(positiveBoxes);
                if (negativeBoxesWidget) negativeBoxesWidget.value = JSON.stringify(negativeBoxes);
            }

            // Mouse events
            canvas.addEventListener("mousedown", (e) => {
                const { x, y } = toCanvasCoords(e);

                if (currentMode === "box") {
                    // For boxes: left = positive, right/Shift = negative
                    const isNegative = e.button === 2 || e.shiftKey;
                    currentPolarity = isNegative ? "negative" : "positive";
                    updateModeIndicator();

                    isDrawingBox = true;
                    boxStartX = x;
                    boxStartY = y;
                } else {
                    // Points: left = positive, right/Shift = negative
                    const isNegative = e.button === 2 || e.shiftKey;
                    const point = { x, y };
                    if (isNegative) {
                        negativePoints.push(point);
                    } else {
                        positivePoints.push(point);
                    }
                    updateWidgets();
                    redraw();
                }

                e.preventDefault();
            });

            canvas.addEventListener("mousemove", (e) => {
                if (!isDrawingBox) return;
                const { x, y } = toCanvasCoords(e);

                redraw();
                ctx.strokeStyle = currentPolarity === "positive" ? "#00ffff" : "#ff0000";
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(boxStartX, boxStartY, x - boxStartX, y - boxStartY);
                ctx.setLineDash([]);
            });

            canvas.addEventListener("mouseup", (e) => {
                if (!isDrawingBox) return;
                const { x, y } = toCanvasCoords(e);

                const box = {
                    x1: Math.min(boxStartX, x),
                    y1: Math.min(boxStartY, y),
                    x2: Math.max(boxStartX, x),
                    y2: Math.max(boxStartY, y)
                };

                const minSize = 1;
                if (Math.abs(box.x2 - box.x1) >= minSize && Math.abs(box.y2 - box.y1) >= minSize) {
                    if (currentPolarity === "positive") {
                        positiveBoxes.push(box);
                    } else {
                        negativeBoxes.push(box);
                    }
                    updateWidgets();
                }

                isDrawingBox = false;
                redraw();
            });

            canvas.addEventListener("contextmenu", (e) => e.preventDefault());

            canvas.addEventListener("mouseenter", () => {
                isMouseOverCanvas = true;
            });

            canvas.addEventListener("mouseleave", () => {
                isMouseOverCanvas = false;
            });

            // Keyboard: only active when mouse over canvas
            document.addEventListener("keydown", (e) => {
                if (!isMouseOverCanvas) return;

                const k = e.key.toLowerCase();

                if (k === "b") {
                    // Toggle between point and box, default positive
                    if (currentMode === "point") {
                        currentMode = "box";
                    } else {
                        currentMode = "point";
                    }
                    currentPolarity = "positive";
                    updateModeIndicator();
                } else if (e.key === "Escape") {
                    // Reset to positive point mode
                    currentMode = "point";
                    currentPolarity = "positive";
                    updateModeIndicator();
                }
            });

            // Clear / Undo
            clearButton.onclick = () => {
                positivePoints = [];
                negativePoints = [];
                positiveBoxes = [];
                negativeBoxes = [];
                updateWidgets();
                redraw();
            };

            undoButton.onclick = () => {
                if (negativePoints.length > 0) {
                    negativePoints.pop();
                } else if (positivePoints.length > 0) {
                    positivePoints.pop();
                } else if (negativeBoxes.length > 0) {
                    negativeBoxes.pop();
                } else if (positiveBoxes.length > 0) {
                    positiveBoxes.pop();
                }
                updateWidgets();
                redraw();
            };

            // Background image loading
            function applyBackgroundFromMessage(message) {
                if (!message || !message.bg_image || !message.bg_image[0]) return;
                const img = new Image();
                img.onload = () => {
                    backgroundImage = img;
                    canvas.width = img.width;
                    canvas.height = img.height;
                    // ensure prompts are in memory before drawing
                    redraw();
                };
                img.src = "data:image/jpeg;base64," + message.bg_image[0];
            }

            // INITIAL LOAD: restore prompts and background, then redraw
            loadFromWidgets();           // <- added
            applyBackgroundFromMessage(node.lastMessage);

            const origOnExecuted = node.onExecuted;
            node.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                loadFromWidgets();       // <- added (in case widgets changed)
                applyBackgroundFromMessage(message);
            };

            return ret;
        };
    }
});
