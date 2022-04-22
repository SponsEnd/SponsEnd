let video_player = document.getElementById("video-player");
let video_submit = document.getElementById("video-submit");

video_submit.addEventListener("click", (e) => {
    e.preventDefault();
    let video_input = document.getElementById("video-input");
    video_player.src = video_input.value;
    console.log(window.location);
    window.location.assign(`${window.location.href}#Video`);
    addNewIframe(video_input.value);
    getValueFromBackend(video_input.value);
});

let addNewIframe = (source) => {
    source = source.split("=")[1];
    document.getElementsByClassName(
        "bubs1"
    )[0].innerHTML = `<iframe width="640" height="470" src="https://www.youtube.com/embed/${source}" id="video-player"></iframe>`;
};

let getValueFromBackend = (source) => {
    source = source.split("=")[1];
    fetch(`http://locahost:5000/sponsor?id=${source}`).then((response) =>
        console.log(response)
    );
};
