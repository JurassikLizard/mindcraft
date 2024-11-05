import { strictFormat } from '../utils/text.js';
import { getCommandDocsAsOpenAI } from '../agent/commands/index.js';
import { snakeToCamel } from '../utils/text.js';

export class Local {
    constructor(model_name, url) {
        this.model_name = model_name;
        this.url = url || 'http://127.0.0.1:11434';
        this.chat_endpoint = '/api/chat';
        this.embedding_endpoint = '/api/embeddings';
    }

    async sendRequest(turns, systemMessage, use_tools=false) {
        let model = this.model_name || 'llama3';
        let messages = strictFormat(turns);
        messages.unshift({role: 'system', content: systemMessage});
        let res = null;
        try {
            console.log(`Awaiting local response... (model: ${model})`)
            let content = {model: model, messages: messages, stream: false};
            if (use_tools)
                content.tools = getCommandDocsAsOpenAI();
            res = await this.send(this.chat_endpoint, content);
            if (res)
                res = res['message']['content'];
            
            // Tool parsing
            if (res['message']['tool_calls'] && res['message']['tool_calls'].length > 0) {
                const function_call = res['message']['tool_calls'][0]['function'];
                res += ' !' + snakeToCamel(function_call["name"]);
                if(function_call["arguments"].length > 0) {
                    res += `(${Object.entries(function_call["arguments"]).map(([k, v], i) => v).join(', ')})`;
                }
            }
        }
        catch (err) {
            if (err.message.toLowerCase().includes('context length') && turns.length > 1) {
                console.log('Context length exceeded, trying again with shorter context.');
                return await sendRequest(turns.slice(1), systemMessage, stop_seq);
            } else {
                console.log(err);
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async embed(text) {
        let model = this.model_name || 'nomic-embed-text';
        let body = {model: model, prompt: text};
        let res = await this.send(this.embedding_endpoint, body);
        return res['embedding']
    }

    async send(endpoint, body) {
        const url = new URL(endpoint, this.url);
        let method = 'POST';
        let headers = new Headers();
        const request = new Request(url, {method, headers, body: JSON.stringify(body)});
        let data = null;
        try {
            const res = await fetch(request);
            if (res.ok) {
                data = await res.json();
            } else {
                throw new Error(`Ollama Status: ${res.status}`);
            }
        } catch (err) {
            console.error('Failed to send Ollama request.');
            console.error(err);
        }
        return data;
    }
}