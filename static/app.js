// 1. Импорт в самом начале
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// --- ОБЩИЕ ЭЛЕМЕНТЫ ---
const fullTextEl = document.getElementById("fullText");   // Поле для жестов
const voiceTextEl = document.getElementById("voiceText"); // НОВОЕ: Поле для голоса
const clearBtn = document.getElementById("clearBtn");     // Кнопка очистки

// --- ЧАСТЬ 1: КАМЕРА И ЖЕСТЫ ---
const toggleBtn = document.getElementById("toggleBtn");
const statusEl = document.getElementById("status");
const currentWordEl = document.getElementById("currentWord");
const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("canvas");
const ctx = canvasEl.getContext("2d");

let videoStream = null;
let timerId = null;
let inFlight = false;
let lastAppendedWord = "";

async function sendFrame() {
  if (!videoStream || inFlight) return;
  inFlight = true;
  try {
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    const dataUrl = canvasEl.toDataURL("image/jpeg", 0.6);

    const resp = await fetch("/api/recognize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: dataUrl }),
    });

    const json = await resp.json();
    const currentWord = json.word || "";
    
    currentWordEl.value = currentWord;
    statusEl.textContent = currentWord ? `Распознано: ${currentWord}` : "Покажите жест...";

    if (currentWord) {
        if (currentWord !== lastAppendedWord) {
            // Добавляем жесты в поле fullText
            fullTextEl.value += (fullTextEl.value ? " " : "") + currentWord;
            lastAppendedWord = currentWord; 
        }
    } else {
        lastAppendedWord = "";
    }
  } catch (err) {
    statusEl.textContent = "Ошибка распознавания жестов";
  } finally {
    inFlight = false;
  }
}

async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    videoEl.srcObject = videoStream;
    await videoEl.play();
    statusEl.textContent = "Камера включена";
    timerId = setInterval(sendFrame, 250);
    toggleBtn.textContent = "Выключить камеру";
  } catch (err) {
    statusEl.textContent = "Нет доступа к камере";
  }
}

function stopCamera() {
  if (timerId) clearInterval(timerId);
  timerId = null;
  if (videoStream) videoStream.getTracks().forEach(t => t.stop());
  videoStream = null;
  videoEl.srcObject = null;
  toggleBtn.textContent = "Включить камеру";
  statusEl.textContent = "Камера выключена";
}

toggleBtn.addEventListener("click", () => videoStream ? stopCamera() : startCamera());

// Кнопка очистки теперь чистит ОБА поля
clearBtn.addEventListener("click", () => {
    fullTextEl.value = "";
    voiceTextEl.value = ""; // Чистим голосовое поле
    lastAppendedWord = "";
});


// --- ЧАСТЬ 2: ГОЛОСОВОЙ ВВОД ---
const voiceStatus = document.getElementById('voice-status');
const recordBtn = document.getElementById('recordBtn');

let transcriber;
let recorder;
let audioChunks = [];

// Инициализация нейросети
(async () => {
    try {
        transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny');
        voiceStatus.innerText = 'Готов';
        recordBtn.innerText = 'Начать говорить';
        recordBtn.disabled = false;
    } catch (e) {
        voiceStatus.innerText = 'Ошибка загрузки модели';
        console.error(e);
    }
})();

recordBtn.onclick = async () => {
    if (recorder && recorder.state === "recording") {
        recorder.stop();
        recordBtn.classList.remove('recording');
        recordBtn.innerText = "Обработка...";
        return;
    }

    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(audioStream);
        audioChunks = [];

        recorder.ondataavailable = e => audioChunks.push(e.data);
        
        recorder.onstop = async () => {
            voiceStatus.innerText = "Распознаю...";
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioContext = new AudioContextClass({ sampleRate: 16000 });
            
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            const result = await transcriber(audioBuffer.getChannelData(0), {
                language: 'russian',
                task: 'transcribe'
            });

            // НОВОЕ: Результат идет в voiceTextEl
            if (result.text && result.text.trim()) {
                const text = result.text.trim();
                voiceTextEl.value += (voiceTextEl.value ? " " : "") + text;
            }
            
            voiceStatus.innerText = "Готово";
            recordBtn.innerText = "Начать говорить";
            // Останавливаем микрофон после записи
            audioStream.getTracks().forEach(t => t.stop());
        };

        recorder.start();
        recordBtn.classList.add('recording');
        recordBtn.innerText = "Стоп";
        voiceStatus.innerText = "Слушаю...";
    } catch (err) {
        voiceStatus.innerText = "Ошибка микрофона";
    }
};