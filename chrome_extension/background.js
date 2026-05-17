chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "ocen-tekst",
        title: "Wylicz ocenę dla tego tekstu",
        contexts: ["selection"]
    });
    
    // Zabezpieczenie: domyślnie ustawiamy gwiazdki przy pierwszej instalacji
    chrome.storage.sync.get(['skala'], function(result) {
        if (!result.skala) {
            chrome.storage.sync.set({ skala: 'gwiazdki' });
        }
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "ocen-tekst") {
        const wybranyTekst = info.selectionText;
        
        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: wybranyTekst })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                wyswietlAlert(tab.id, "Błąd: " + data.error);
                return;
            } 
            if (data.message) {
                wyswietlAlert(tab.id, data.message);
                return;
            }

            // ODCZYT USTAWIENIA SKALI I KONWERSJA
            chrome.storage.sync.get(['skala'], function(result) {
                let wiadomosc = "";
                let ocena = data.rating; // Wartość między 1.0 a 5.0

                if (result.skala === 'punkty') {
                    // Mnożymy wynik x2, aby uzyskać skalę na 10 punktów
                    let punkty = (ocena * 2).toFixed(1);
                    
                    // Inteligentna logika dla skrajnie słabych tekstów
                    if (ocena <= 1.25) {
                        wiadomosc = `Przewidywana ocena: 0/10 - ${punkty}/10\n(Surowy wynik: ${punkty}/10)\n\n💡 UWAGA: Tekst o skrajnie negatywnym wydźwięku. Ponieważ sztuczna inteligencja była trenowana na skali 1-5, minimalny matematyczny wynik to 2.0/10 (1 gwiazdka). W rzeczywistości oznacza to absolutne dno skali, stąd rzeczywista ocena może sięgać nawet wyniku 0/10.`;
                    } else {
                        wiadomosc = `Przewidywana ocena dla tego posta to: ${punkty} / 10`;
                    }
                } else {
                    wiadomosc = `Przewidywana ocena dla tego posta to: ${ocena} ⭐`;
                }

                wyswietlAlert(tab.id, wiadomosc);
            });
        })
        .catch(error => {
            wyswietlAlert(tab.id, "Błąd połączenia. Upewnij się, że api_serwer.py jest włączony w tle!");
        });
    }
});

function wyswietlAlert(tabId, wiadomosc) {
    chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: (msg) => alert(msg),
        args: [wiadomosc]
    });
}