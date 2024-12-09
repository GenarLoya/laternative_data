import React from "react";
import { useState } from "react";
import { Toaster, toast } from "sonner";

export const Predictor: React.FC = () => {
    const [email, setEmail] = useState("");

    const predict = async () => {
        const idN = toast.loading("Procesando el correo");

        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                text: email,
            }),
        });
        const data = await response.json();

        if (data.error) {
            toast.error(data.error, { id: idN });
        } else {
            if (data.prediction === "spam") {
                toast.error("El correo es spam", { id: idN });
            } else {
                toast.success("E Correo no es spam", { id: idN });
            }
        }
    };

    return (
        <div className="w-full flex flex-col items-center justify-center">
            <Toaster richColors={true} position="top-center" />
            <textarea
                id="email"
                value={email}
                rows={15}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-2 border-2 border-slate-900 rounded-lg"
                placeholder="Enter your email text here..."
            ></textarea>
            <br />
            <div className="flex justify-between">
                <button
                    className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mx-auto"
                    onClick={predict}>Predecir</button>
                <p id="result"
                    className="text-center text-xl font-semibold text-red-500"
                ></p>
            </div>
        </div>
    );
};