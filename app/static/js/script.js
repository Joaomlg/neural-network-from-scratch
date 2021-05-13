var defaultBoard = new DrawingBoard.Board('default-board', {
  color: "#000",
  size: 20,
  webStorage: false,
  controls: [
    { 
      DrawingMode: { pencil:false, eraser:false, filler: false } 
    },
    { 
      Navigation: { back: false, forward:false} 
    }
  ],
  controlsPosition: 'right',
  enlargeYourContainer: true
});

defaultBoard.ev.bind('board:reset', () => {
  $('#result').html('Result: Cleared');
})

async function predict() {
  const canvas = defaultBoard.canvas;
  const data = await processImage(canvas, 28, 28);
  $.ajax({
    url: '/predict',
    data: JSON.stringify(data),
    contentType: 'application/json; charset=utf-8',
    type: 'POST',
    success: response => {
      const data = JSON.parse(response);
      $('#result').html(`Result: ${data.result}`);
    },
    error: error => {
      console.log(error);
      $('#result').html(`Result: ${Error}`);
    }
  })
}

function processImage(canvas, WIDTH, HEIGHT) {
  return new Promise((resolve) => {
    let img = new Image();
    img.src = canvas.toDataURL();
    img.onload = () => {
      let newCanvas = document.createElement('canvas');
      let ctx = newCanvas.getContext('2d');

      let width = img.width
      let height = img.height

      if (width > height) {
        if (width > WIDTH) {
          height *= WIDTH / width
          width = WIDTH
        }
      } else {
        if (height > HEIGHT) {
          width *= HEIGHT / height
          height = HEIGHT
        }
      }

      newCanvas.width = width
      newCanvas.height = height
      ctx.drawImage(img, 0, 0, width, height)

      ctx.globalCompositeOperation = 'difference';
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, width, height);

      let data = ctx.getImageData(0, 0, width, height).data;
      let grayscale = [];

      for (let i=0; i<data.length; i+=4) {
        let avg = (data[i] + data[i+1] + data[i+2]) / 3;
        grayscale.push(avg);
      }

      resolve(grayscale)
    }
  })
}
