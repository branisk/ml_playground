setInterval("MathJax.Hub.Queue(['Typeset',MathJax.Hub])",1000);

function updateFontSize() {
    var element = document.getElementById("objective-text");
    if (element) {
        element.style.fontSize = "9px";
    }
}

setInterval(updateFontSize, 1000);

function attachAccordionListener() {
    var accordion = document.getElementById('info-accordion');
    if (accordion) {
        accordion.addEventListener('shown.bs.collapse', function () {
            updateFontSize();
        });
        return true;
    }
    return false;
}

function addObserverIfNotExists() {
    if (!attachAccordionListener()) {
        var observer = new MutationObserver(function(mutations) {
            if (attachAccordionListener()) {
                observer.disconnect();
            }
        });

        observer.observe(document.documentElement, {
            childList: true,
            subtree: true
        });
    }
}

document.addEventListener('DOMContentLoaded', addObserverIfNotExists);
