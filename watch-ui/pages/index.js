import axios from 'axios';
import Head from 'next/head';
import { useEffect, useState } from 'react';

export default function Home() {
  const [gestureName, setGestureName] = useState('');

  useEffect(() => {
    getGesture();
    // setInterval(() => {
    //   getGesture();
    // }, 1000);
  });

  const getGesture = async () => {
    const response = await axios.get('http://localhost:3000/api/gestures');
    const data = await response.data.data.gestures.slice(-1)[0];
    setGestureName(data.motionName);
  };
  return (
    <div>
      <Head>
        <title>Gesture Prediction</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <div className="main__display">
          <div className="main__card">
            <div className="main_text">
              <h4>Predicted Gesture</h4>
              <p>{gestureName}</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}