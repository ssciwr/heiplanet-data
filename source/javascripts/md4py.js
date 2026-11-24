// hide the first line in TOC of generated HTML for each module
document.querySelectorAll('a[href^="#heiplanet_data."]').forEach(el => {
    // get the part after "#onehealth_data_backend."
    const suffix = el.getAttribute('href').slice('#heiplanet_data.'.length);
    // hide only if suffix does not contain a dot
    if (!suffix.includes('.')) {
        el.style.display = 'none';
    }
});