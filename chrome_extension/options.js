// Po otwarciu opcji, zaznaczamy to, co było wcześniej wybrane
chrome.storage.sync.get(['skala'], function(result) {
    if (result.skala === 'punkty') {
        document.getElementById('punkty').checked = true;
    } else {
        document.getElementById('gwiazdki').checked = true;
    }
});

// Zapis po kliknięciu przycisku
document.getElementById('zapisz').addEventListener('click', () => {
    const wybranaSkala = document.querySelector('input[name="skala"]:checked').value;
    
    chrome.storage.sync.set({ skala: wybranaSkala }, () => {
        const status = document.getElementById('status');
        status.textContent = 'Ustawienia zapisane pomyślnie!';
        setTimeout(() => { status.textContent = ''; }, 2500);
    });
});