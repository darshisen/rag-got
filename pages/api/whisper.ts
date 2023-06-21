const FormData = require('form-data');
import { withFileUpload } from 'next-multiparty';
import { createReadStream } from 'fs';
import axios from 'axios'; // Import axios

export const config = {
  api: {
    bodyParser: false,
  },
};

export default withFileUpload(async (req, res) => {
  const file = req.file;
  if (!file) {
    res.status(400).send('No file uploaded');
    return;
  }
  console.log(file)

  // Create form data
  const formData = new FormData();
  formData.append('file', createReadStream(file.filepath), 'audio.wav');
  formData.append('model', 'whisper-1');
  
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/audio/transcriptions',
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        },
      }
    );

    const { text, error } = response.data;
    if (response.status === 200) {
      res.status(200).json({ text: text });
    } else {
      console.log('OPEN AI ERROR:');
      console.log(error.message);
      res.status(400).send(new Error());
    }
  } catch (error) {
    console.error(error);
    res.status(500).send('An error occurred');
  }
});

