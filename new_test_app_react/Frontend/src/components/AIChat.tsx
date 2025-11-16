"use client";

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
    Copy,
    Pencil,
    Bot,
    UserCircle2,
} from "lucide-react";
import { cn } from "@/lib/utils"; // if using shadcn utils
import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';

interface AiChatProps {
    model: string;
    isClear: boolean;
    modelName: string;
    businessData: any[];
    models: any[];
    userData: any[];
    lob: any[];
    onView: () => void;
    dateRange: any[];
    cards: { title: string; description: string }[];
}

interface Message {
    text: string;
    sender: "user" | "server";
    isLink?: boolean;
}

const copyToClipboardText = async (text: string): Promise<boolean> => {
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            // fallback for insecure context (like http or older browsers)
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed"; // avoid scrolling to bottom
            textArea.style.opacity = "0";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
        }
        return true;
    } catch (err) {
        console.error("Failed to copy text: ", err);
        return false;
    }
};


const AIChat: React.FC<AiChatProps> = ({
    model,
    isClear,
    modelName,
    businessData,
    models,
    onView,
    userData,
    lob,
    dateRange,
    cards,
}) => {
    const [message, setMessage] = useState("");
    const [messages, setMessages] = useState<Message[]>(localStorage.getItem("ai_messages") ? JSON.parse(localStorage.getItem("ai_messages") || "[]") : []);
    const [serverResponse, setServerResponse] = useState({
        loading: false,
        data: null as any,
        err: null as any,
    });
    const [selectedCard, setSelectedCard] = useState<number>(-1);
    const [isTextHover, setTextHover] = useState(false);
    const [textHoverKey, setTextHoverKey] = useState<number | null>(null);
    const [isFilter, setIsFilter] = useState(false);
    const [isLink, setIsLink] = useState(false);
    const [aiFilterTemp, setAiFilterTemp1] = useState<string | false>(false);


    const chatEndRef = useRef<HTMLDivElement>(null);

    const WEBAPPAPIURL = `${AppConfig.API_URL}/`;

    const onMessageChange = (e) => {
        setMessage(e.target.value);
    };


    function convertBoldMarkdownToHtml(text: string) {
        return text
            .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>") // bold
            .replace(/\n/g, "<br/>");               // line breaks
    }

    // Scroll to the latest message when messages update
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const textareaRef = useRef<HTMLTextAreaElement | null>(null);

    // Focus when component first mounts
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.focus();
        }
    }, []);

    // Focus again whenever message is cleared
    useEffect(() => {
        if (textareaRef.current && message === "" && !serverResponse.loading) {
            textareaRef.current.focus();
        }
    }, [message, serverResponse.loading]);

    useEffect(() => {
        if (messages && messages.length) {
            localStorage.setItem("ai_messages", JSON.stringify(messages));
        }
        if (model) {
            const oldModel = localStorage.getItem("ai_model");
            if (oldModel !== model) {
                localStorage.setItem("ai_messages", JSON.stringify([]));
                setMessages([]);
                localStorage.setItem("ai_model", model);
            } else {
                localStorage.setItem("ai_model", model);
            }
        }
    }, [JSON.stringify(messages), model]);

    useEffect(() => {
        if (isClear) {
            setMessages([]);
            setSelectedCard(-1);
        }
    }, [isClear]);

    const isJsonString = (str: string) => {
        try {
            JSON.parse(str);
            return true;
        } catch {
            return false;
        }
    };

    const handleRecieveMessage = (msg: string) => {
        if (message) {
            setServerResponse({ loading: true, data: null, err: null });
            setIsFilter(false);
            setIsLink(false);
            const tempMessage: Message = { text: "Processing...", sender: "server" };
            setMessages((prev) => [...prev, tempMessage]);

            const params = new URLSearchParams({
                user: JSON.stringify({ id: userData?.sub, name: userData?.name }),
                company: AuthService.getCompanyId().toString(),
                business_unit: JSON.stringify({ id: businessData?.id, name: businessData?.display_name }),
                lob: JSON.stringify(lob),
                module_name: model,
                // model_name: JSON.stringify(models),
                date_range: JSON.stringify(dateRange),
                token: AuthService.getAccessToken(),
                prompt: msg,
            });


            axios
                .get(
                    `${WEBAPPAPIURL}webhook/94a9f646-5ab4-41a1-a109-c64932b24e9b?${params.toString()}`,
                    { headers: { portalDomain: window.location.origin } }
                )
                .then((response) => {
                    const result = response.data?.length
                        ? response.data[0]
                        : false;
                    setMessages((prev) =>
                        prev.map((m, i) =>
                            i === prev.length - 1
                                ? {
                                    text: result?.output
                                        ? (() => {
                                            try {
                                                if (isJsonString(result.output)) {
                                                    const parsed = JSON.parse(result.output);
                                                    if (parsed?.type) {
                                                        setIsLink(true);
                                                    }
                                                    return parsed?.response_text || result.output;
                                                }
                                                return result.output;
                                            } catch (error) {
                                                console.log('Error parsing JSON:', error);
                                                setIsLink(false);
                                                return result.output;
                                            }
                                        })()
                                        : "Sorry unable to process your request",
                                    sender: "server",
                                    isLink: result?.output
                                        ? (() => {
                                            try {
                                                if (isJsonString(result.output)) {
                                                    const parsed = JSON.parse(result.output);
                                                    if (parsed?.type) {
                                                        return true
                                                    }
                                                    return false;
                                                }
                                                return false;
                                            } catch (error) {
                                                return false;
                                            }
                                        })()
                                        : false,
                                }
                                : m
                        )
                    );
                    setServerResponse({ loading: false, data: response.data, err: null });
                })
                .catch(() => {
                    setServerResponse({ loading: false, data: null, err: "Error" });
                    setIsFilter(false);
                    setIsLink(false);
                    setMessages((prev) =>
                        prev.map((m, i) =>
                            i === prev.length - 1
                                ? { text: "Sorry, Network Error.", sender: "server" }
                                : m
                        )
                    );
                });
        }
    };

    const handleSendMessage = () => {
        if (!message.trim()) return;
        const userMessage: Message = { text: message, sender: "user" };
        setMessages((prev) => [...prev, userMessage]);
        handleRecieveMessage(message);
        setMessage("");
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            setSelectedCard(-1);
            handleSendMessage();
        }
    };

    useEffect(() => {
        if (isClear) {
            setMessages([]);
            setSelectedCard(-1);
        }
    }, [isClear]);

    return (
        <div className="flex flex-col h-full">
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 hidden-scrollbar">
                {messages.length === 0 && (
                    <div className="text-center text-foreground">
                        <h5 className="font-semibold">
                            Hi <span className="text-blue-500">{userData?.name}!</span>
                        </h5>
                        <p className="text-sm text-foreground opacity-60">How can I help you with {modelName}?</p>
                    </div>
                )}

                {/* Cards */}
                {messages.length === 0 &&
                    cards.map((card, index) => (
                        <div
                            key={index}
                            onClick={() => {
                                setSelectedCard(index);
                                setMessage(card.description);
                            }}
                            className={cn(
                                "border rounded-lg p-3 mb-2 mt-3 cursor-pointer transition hover:bg-muted",
                                selectedCard === index && "bg-muted"
                            )}
                        >
                            <div className="flex items-start gap-2">
                                <div className="flex-shrink-0 w-6 h-6 text-primary">
                                    <Bot className="w-full h-full" />
                                </div>
                                <div>
                                    <p className="font-semibold text-foreground">{card.title}</p>
                                    <p className="text-sm opacity-70 text-foreground">{card.description}</p>
                                </div>
                            </div>
                        </div>
                    ))}

                {/* Messages */}
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className="mb-4 relative group"
                    >
                        <div
                            className={cn(
                                "flex items-start",
                                msg.sender === "user" ? "justify-end" : "justify-start"
                            )}
                        >
                            {msg.sender === "server" && (
                                <div className="flex-shrink-0 mt-1 mr-2">
                                    <Bot className="w-6 h-6 text-primary" />
                                </div>
                            )}

                            {msg.sender === "server" && (
                                <div className="max-w-[75%]">
                                    <div
                                        className={cn(
                                            "relative px-3 py-2 rounded-lg max-w-[100%] text-sm",
                                            "bg-muted text-foreground"
                                        )}
                                        dangerouslySetInnerHTML={{ __html: convertBoldMarkdownToHtml(msg.text) }}
                                    />
                                    {msg.isLink && (messages.length === (index + 1)) && (
                                        <p onClick={() => onView()} className="mt-2 text-sm cursor-pointer text-blue-500 text-underline">
                                            View Changes
                                        </p>
                                    )}
                                </div>
                            )}

                            {msg.sender === "user" && (
                                <div className="relative flex-shrink max-w-[75%] flex flex-col">
                                    {/* Message bubble */}
                                    <div
                                        className={cn(
                                            "px-3 py-2 rounded-lg text-sm",
                                            "bg-primary text-primary-foreground"
                                        )}
                                    >
                                        {msg.text}
                                    </div>

                                    {/* Hover buttons below the message */}
                                    <div
                                        className="flex gap-2 mt-1 opacity-0 group-hover:opacity-90 pointer-events-none ml-auto group-hover:pointer-events-auto transition-opacity"
                                    >
                                        <button
                                            onClick={() => copyToClipboardText(msg.text)}
                                            className="p-1 rounded-md bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700"
                                        >
                                            <Copy className="w-4 h-4 text-gray-700 dark:text-gray-200" />
                                        </button>
                                        <button
                                            onClick={() => setMessage(msg.text)}
                                            className="p-1 rounded-md bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700"
                                        >
                                            <Pencil className="w-4 h-4 text-gray-700 dark:text-gray-200" />
                                        </button>
                                    </div>
                                </div>
                            )}


                            {msg.sender === "user" && (
                                <div className="flex-shrink-0 mt-1 ml-2">
                                    <UserCircle2 className="w-5 h-5 text-muted-foreground" />
                                </div>
                            )}
                        </div>
                    </div>
                ))}



                <div ref={chatEndRef} />
            </div>

            {/* Sticky Input Area */}
            <div className="p-2 border-t bg-background">
                <textarea
                    ref={textareaRef}
                    value={message}
                    disabled={serverResponse.loading}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyDown={handleKeyDown}
                    rows={2}
                    autoFocus
                    placeholder="Enter your prompt here..."
                    className="w-full resize-none rounded-md p-2 text-sm 
               bg-background text-foreground border 
               border-input shadow-sm 
               focus:outline-none focus:ring-2 focus:ring-primary"
                />
                <p className="text-xs text-muted-foreground mt-2 mb-1 text-center">
                    Ask Gauri AI (Beta) can make mistakes.
                </p>
            </div>

        </div>

    );
};

export default AIChat;
