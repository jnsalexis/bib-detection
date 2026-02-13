// État initial des scripts
let scriptsStatus = {
    stream: false,
    process_images: false
};

// Initialiser l'état depuis les données du serveur
function initScriptsStatus(initialStatus) {
    if (initialStatus) {
        scriptsStatus.stream = initialStatus.stream || false;
        scriptsStatus.process_images = initialStatus.process_images || false;
    }
    updateScriptsUI();
}

// Mettre à jour l'interface selon l'état
function updateScriptsUI() {
    updateScriptUI('stream');
    updateScriptUI('process_images');
}

function updateScriptUI(scriptName) {
    const isRunning = scriptsStatus[scriptName];
    let startBtnId, stopBtnId;
    
    if (scriptName === 'process_images') {
        startBtnId = 'btn-start-process';
        stopBtnId = 'btn-stop-process';
    } else {
        startBtnId = 'btn-start-stream';
        stopBtnId = 'btn-stop-stream';
    }
    
    const startBtn = document.getElementById(startBtnId);
    const stopBtn = document.getElementById(stopBtnId);
    const statusDiv = document.getElementById(`status-${scriptName}`);
    
    if (startBtn && stopBtn && statusDiv) {
        if (isRunning) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            statusDiv.innerHTML = '<span style="color: #27ae60;">●</span> En cours d\'exécution';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusDiv.innerHTML = '<span style="color: #95a5a6;">○</span> Arrêté';
        }
    }
}

async function startScript(scriptName) {
    try {
        const response = await fetch(`/api/scripts/start/${scriptName}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            scriptsStatus[scriptName] = true;
            updateScriptUI(scriptName);
            alert(`${scriptName}.py démarré avec succès`);
        } else {
            alert('Erreur: ' + result.error);
        }
    } catch (error) {
        alert('Erreur lors du démarrage: ' + error.message);
    }
}

async function stopScript(scriptName) {
    if (!confirm(`Êtes-vous sûr de vouloir arrêter ${scriptName}.py ?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/scripts/stop/${scriptName}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            scriptsStatus[scriptName] = false;
            updateScriptUI(scriptName);
            alert(`${scriptName}.py arrêté avec succès`);
        } else {
            alert('Erreur: ' + result.error);
        }
    } catch (error) {
        alert('Erreur lors de l\'arrêt: ' + error.message);
    }
}

// Vérifier l'état des scripts toutes les 5 secondes
async function checkScriptsStatus() {
    try {
        const response = await fetch('/api/scripts/status');
        const result = await response.json();
        
        if (result.status) {
            scriptsStatus.stream = result.status.stream.running;
            scriptsStatus.process_images = result.status.process_images.running;
            updateScriptsUI();
        }
    } catch (error) {
        console.error('Erreur lors de la vérification du statut:', error);
    }
}

async function deleteDossard(id) {
    if (!confirm(`Êtes-vous sûr de vouloir supprimer le dossard #${id} ?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/dossards/${id}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('Dossard supprimé avec succès !');
            location.reload();
        } else {
            alert('Erreur: ' + result.error);
        }
    } catch (error) {
        alert('Erreur lors de la suppression: ' + error.message);
    }
}

// Initialiser le polling du statut si on est sur la page d'accueil
if (document.getElementById('btn-start-stream')) {
    // Vérifier le statut toutes les 5 secondes
    setInterval(checkScriptsStatus, 5000);
}
